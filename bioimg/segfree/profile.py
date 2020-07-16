#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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

class SegfreeProfiler:
    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        tile_size : tuple
        km_pixel : KMeans object for pixels (color compression)
        km_block : KMeans object for tiles (blocks)
        km_supblock : KMeans object for superblocks
        '''
        self.tile_size = kwargs.get('tile_size', None)
        self.n_block_types = kwargs.get('n_block_types', 50)
        self.n_supblock_types = kwargs.get('n_supblock_types', 30)
        
        # these are initialized with 'None'
        self.km_pixel = None
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

    def fit(self, imgs, n_init=50, random_state=1307):
        img_tiles = self.tile_images(imgs)
        # nbits = np.unique(np.stack(img_tiles))
        blockfeats = [get_blockfeats(t) for t in img_tiles]
        blockdf = pd.concat(blockfeats).fillna(0)
        self.km_block = KMeans(n_clusters=self.n_block_types,
                               n_init=n_init,
                               random_state=random_state).fit(blockdf)
        cols = blockdf.columns.values
        grid_shape = tuple(int(x / y) for x,y in zip(imgs[0].shape, self.tile_size))
        supblocks = [get_supblocks(bf,
                                   km_block=self.km_block,
                                   cols=cols,
                                   grid_shape=grid_shape) for bf in blockfeats]
        self.km_supblock = KMeans(n_clusters=self.n_supblock_types,
                                  n_init=n_init,
                                  random_state=random_state).fit(pd.concat(supblocks).fillna(0))
        return self

    def transform(self, imgs):
        img_tiles = self.tile_images(imgs)
        #nbits = np.unique(np.stack(img_tiles))
        blockfeats = [get_blockfeats(t) for t in img_tiles]
        blockdf = pd.concat(blockfeats).fillna(0)
        cols = blockdf.columns.values
        grid_shape = tuple(int(x / y) for x,y in zip(imgs[0].shape, self.tile_size))
        pixel_mean = pd.concat([get_blocktype(t) for t in img_tiles]).fillna(0).reset_index(drop=True)
        pixel_mean.columns = ['-'.join(['pixel', str(col)])
                           for col in pixel_mean.columns.values]
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
