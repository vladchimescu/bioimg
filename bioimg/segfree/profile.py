#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.util import view_as_blocks, view_as_windows
from skimage.util import img_as_ubyte

def get_blocktype(a):
    block_type = dict(zip(*np.unique(a, return_counts=True)))
    return pd.DataFrame([block_type]) / a.size

def get_blockfeats(blocks):
    mask = np.array([(bl != 0).sum() > 0.5 * bl.size for bl in blocks])
    blockfeats = pd.concat([get_blocktype(bl) for bl in blocks[mask]])
    blockfeats.index = np.where(mask)[0]
    return blockfeats

def get_supblocks(bf, km_block, cols, grid_shape, window_shape=3, thresh=0.5):
    # plus one for background
    n_sb = len(np.unique(km_block.labels_)) + 1
    img_blocked = np.zeros(grid_shape[0] * grid_shape[1])
    # make sure has the same columns as all other blocks
    bf = bf.reindex(columns=cols).fillna(0)
    # only if index is set (foreground blocks)
    img_blocked[bf.index] = km_block.predict(bf) + 1
    img_blocked = img_blocked.reshape(grid_shape)
    supblocks = view_as_windows(img_blocked,
                                window_shape=window_shape).reshape(-1,window_shape,
                                                                   window_shape)
    mid = np.ceil(window_shape/2).astype(int) - 1
    fgr_supblocks = np.stack([sb for sb in supblocks if sb[mid,mid]])
    return pd.concat([get_blocktype(sb) for sb in fgr_supblocks])

def get_color_supblocks(img, window_shape=3):
    supblocks = view_as_windows(img,
                                    window_shape=window_shape).reshape(-1,window_shape, window_shape)
    return pd.concat([get_blocktype(sb) for sb in supblocks])

def flatten_tiles(blocks):
    return np.array([block.ravel() for block in blocks])

class SegfreeProfiler:
    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        tile_size : tuple
        pca : PCA object for dimensionality reduction
        km_block : KMeans object for tiles (blocks)
        km_supblock : KMeans object for superblocks
        '''
        self.tile_size = kwargs.get('tile_size', None)
        self.n_block_types = kwargs.get('n_block_types', 50)
        self.n_supblock_types = kwargs.get('n_supblock_types', 30)
        self.n_components = kwargs.get('n_components', 50)
        self.n_subset = kwargs.get('n_subset', 10000)
        
        # these are initialized with 'None'
        self.pca = None
        self.km_block = None
        self.km_supblock = None

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def set_param(self, **kwargs):
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

    def tile_images(self, imgs):
        h, w = imgs[0].shape
        height, width = self.tile_size
        # clip if the tile size doesn't match image height
        if h % height != 0:
            imgs = [img[0:-(h % height),:] for img in imgs]
        # clip if the tile size doesn't match image width
        if w % width != 0:
            imgs = [img[:,0:-(w % width)] for img in imgs]
        return [view_as_blocks(img,
                               block_shape=self.tile_size).reshape(-1, *self.tile_size) for img in imgs]

    def tile_color_images(self, imgs):
        h, w, nchan = imgs[0].shape
        height, width = self.tile_size
        # clip if the tile size doesn't match image height
        if h % height != 0:
            imgs = [img[0:-(h % height),:,:] for img in imgs]
        # clip if the tile size doesn't match image width
        if w % width != 0:
            imgs = [img[:,0:-(w % width),:] for img in imgs]
        return [view_as_blocks(img, block_shape=(height, width, nchan)).reshape(-1, height, width, nchan) for img in imgs]

    def _fit_single_channel(self, imgs, n_init,
                            random_state,
                            transform=False):
        img_tiles = self.tile_images(imgs)
        print("Estimating tile properties")
        blockfeats = [get_blockfeats(t) for t in img_tiles]
        blockdf = pd.concat(blockfeats).fillna(0)
        cols = blockdf.columns.values
        print("Running k-means on tiles")
        self.km_block = KMeans(n_clusters=self.n_block_types,
                               n_init=n_init,
                               random_state=random_state).fit(blockdf)
        
        grid_shape = tuple(int(x / y) for x,y in zip(imgs[0].shape, self.tile_size))
        supblocks = [get_supblocks(bf,
                                   km_block=self.km_block,
                                   cols=cols,
                                   grid_shape=grid_shape) for bf in blockfeats]
        print("Running k-means on superblocks")
        self.km_supblock = KMeans(n_clusters=self.n_supblock_types,
                                  n_init=n_init,
                                  random_state=random_state).fit(pd.concat(supblocks).fillna(0))
        if transform:
            return self._transform_single_channel(imgs, supblocks)
        print("Done")

    def _fit_multichannel(self, imgs, n_init,
                          random_state,
                          transform=False):
        img_tiles = self.tile_color_images(imgs)
        Xtrain = np.concatenate([flatten_tiles(t) for t in img_tiles])
        print("Running PCA on tiles")
        self.pca = PCA(n_components=self.n_components,
                       svd_solver='randomized',
                       whiten=True,
                       random_state=random_state).fit(Xtrain)
        blockdf = self.pca.transform(Xtrain)
        np.random.seed(random_state)
        subset = np.random.choice(range(blockdf.shape[0]), size=self.n_subset)
        print("Running k-means on tiles")
        self.km_block = KMeans(n_clusters=self.n_block_types,
                               n_init=n_init,
                               random_state=random_state).fit(blockdf[subset,:])

        grid_shape = tuple(int(x / y) for x,y in zip(imgs[0].shape, self.tile_size))
        img_blocked = self.km_block.predict(blockdf).reshape(-1, *grid_shape)
        supblocks = [get_color_supblocks(img_blocked[i]) for i in range(img_blocked.shape[0])]
        print("Running k-means on superblocks")
        self.km_supblock = KMeans(n_clusters=self.n_supblock_types,
                                  n_init=n_init,
                                  random_state=random_state).fit(pd.concat(supblocks).fillna(0))
        if transform:
            return self._transform_multichannel(imgs, supblocks)
        print("Done")
        
        

    def fit(self, imgs, n_init=50, random_state=1307):
        if imgs[0].ndim == 2:
            print("Fitting model for greyscale images")
            self._fit_single_channel(imgs=imgs, n_init=n_init, random_state=random_state)
        if imgs[0].ndim == 3:
            print("Fitting model for multichannel images")
            self._fit_multichannel(imgs=imgs, n_init=n_init, random_state=random_state)
        return self

    def fit_transform(self, imgs, n_init=50, random_state=1307):
        if imgs[0].ndim == 2:
            print("Fitting model for greyscale images")
            return self._fit_single_channel(imgs=imgs,
                                     n_init=n_init,
                                     random_state=random_state,
                                     transform=True)
        if imgs[0].ndim == 3:
            print("Fitting model for multichannel images")
            return self._fit_multichannel(imgs=imgs,
                                          n_init=n_init,
                                          random_state=random_state,
                                          transform=True)

    def _transform_single_channel(self, imgs,
                                  supblocks=None):
        img_tiles = self.tile_images(imgs)           
        pixel_mean = pd.concat([get_blocktype(t) for t in img_tiles]).fillna(0).reset_index(drop=True)
        pixel_mean.columns = ['-'.join(['pixel', str(col)])
                           for col in pixel_mean.columns.values]
        grid_shape = tuple(int(x / y) for x,y in zip(imgs[0].shape, self.tile_size))
        if supblocks is None:
            blockfeats = [get_blockfeats(t) for t in img_tiles]
            blockdf = pd.concat(blockfeats).fillna(0)
            cols = blockdf.columns.values
            supblocks = [get_supblocks(bf,
                                       km_block=self.km_block,
                                       cols=cols,
                                       grid_shape=grid_shape) for bf in blockfeats]
        sup_cols = pd.concat(supblocks).columns.values.astype(np.int)
        block_mean = pd.concat([bf.reindex(columns=sup_cols).fillna(0).agg('mean') for bf in supblocks], axis=1).T
        block_mean.columns = ['-'.join(['block', str(col)])
                           for col in block_mean.columns.values]
        supblock_mean = pd.concat([get_blocktype(self.km_supblock.predict(sbf.reindex(columns=sup_cols).fillna(0)) + 1)
                       for sbf in supblocks]).reset_index(drop=True)
        supblock_mean = supblock_mean.fillna(0)
        supblock_mean.columns = ['-'.join(['superblock', str(col)])
           for col in supblock_mean.columns.values]
        img_prof = pd.concat([supblock_mean, block_mean,  pixel_mean], axis=1)
        return img_prof

    def _transform_multichannel(self, imgs,
                                supblocks=None):
        img_tiles = self.tile_color_images(imgs)
        if supblocks is None:
            Xtest = np.concatenate([flatten_tiles(t) for t in img_tiles])
            blockdf = self.pca.transform(Xtest)
            grid_shape = tuple(int(x / y) for x,y in zip(imgs[0].shape, self.tile_size))
            img_blocked = self.km_block.predict(blockdf).reshape(-1, *grid_shape)
            supblocks = [get_color_supblocks(img_blocked[i]) for i in range(img_blocked.shape[0])]
        sup_cols = pd.concat(supblocks).columns.values.astype(np.int)
        block_mean = pd.concat([bf.reindex(columns=sup_cols).fillna(0).agg('mean') for bf in supblocks], axis=1).T
        block_mean.columns = ['-'.join(['block', str(col)])
                           for col in block_mean.columns.values]
        supblock_mean = pd.concat([get_blocktype(self.km_supblock.predict(sbf.reindex(columns=sup_cols).fillna(0)) + 1)
                       for sbf in supblocks]).reset_index(drop=True)
        supblock_mean = supblock_mean.fillna(0)
        supblock_mean.columns = ['-'.join(['superblock', str(col)])
           for col in supblock_mean.columns.values]
        img_prof = pd.concat([supblock_mean, block_mean], axis=1)
        return img_prof


    def transform(self, imgs):
        if imgs[0].ndim == 2:
            return self._transform_single_channel(imgs)
        if imgs[0].ndim == 3:
            return self._transform_multichannel(imgs)
