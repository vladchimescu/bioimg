#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.util import view_as_blocks, view_as_windows
from skimage.util import img_as_ubyte


def tile_images(imgs, tile_size):
    '''Tile images
       --------------------------------------
       Grid images into blocks of specified width
       and height and return the list of image tiles.
       Supports greyscale, color and 3D greyscale images

       Parameters
       ----------
       imgs : list or array of images
       tile_size : tuple
       
       Returns
       -------
       list : list of tiled images
    '''
    height, width = tile_size
    # if greyscale image
    if imgs[0].ndim == 2:
        h, w = imgs[0].shape
        # clip if the tile size doesn't match image height
        if h % height != 0:
            imgs = [img[0:-(h % height),:] for img in imgs]
        # clip if the tile size doesn't match image width
        if w % width != 0:
            imgs = [img[:,0:-(w % width)] for img in imgs]
        block_shape = (height, width)
    # if multichannel or 3D image
    if imgs[0].ndim == 3:
        h, w, nchan = imgs[0].shape
        # clip if the tile size doesn't match image height
        if h % height != 0:
            imgs = [img[0:-(h % height),:,:] for img in imgs]
        # clip if the tile size doesn't match image width
        if w % width != 0:
            imgs = [img[:,0:-(w % width),:] for img in imgs]
        block_shape = (height, width, nchan)
    if imgs[0].ndim > 3:
        raise TypeError("Only 2D and 3D image arrays are supported")
    return [view_as_blocks(img, block_shape=block_shape).reshape(-1, *block_shape) for img in imgs]

def get_block_counts(a):
    '''Counts frequency of each element in an image block
       --------------------------------------------------
       Computes occurence frequency of block elements,
       such as pixel values (for int-valued images) or
       block types in a superblock

       Parameters
       ----------
       a : array
           Image block

       Returns
       -------
       df : DataFrame
           DataFrame with frequencies of each element.
           Columns indicate bins / levels
    '''
    block_type = dict(zip(*np.unique(a, return_counts=True)))
    return pd.DataFrame([block_type]) / a.size

def get_greyscale_blockfeats(blocks):
    '''Compute features for greyscale image blocks
       -------------------------------------------
       For greyscale blocks, the features are 
       occurence frequencies of individual bits
       (images are assumed to be of 8 or 16-bit integer type)
    
       Parameters
       ----------
       blocks : list-like
           List or array of greyscale image blocks

       Returns
       -------
       df : DataFrame
           DataFrame with blocks in rows and
           block features in columns
    '''
    mask = np.array([(bl != 0).sum() > 0.5 * bl.size for bl in blocks])
    blockfeats = pd.concat([get_block_counts(bl) for bl in blocks[mask]])
    blockfeats.index = np.where(mask)[0]
    return blockfeats

def get_block_types(bf, km_block, cols, grid_shape):
    '''Cluster greyscale image blocks based on features
       ------------------------------------------------
       Use the pre-trained KMeans object to return
       cluster labels of each block in a greyscale image

       Parameters
       ----------
       bf : DataFrame with block features
           Each block is characterized by bit frequencies
       km_block : KMeans object
           KMeans model trained on greyscale image blocks
       cols : array
           greyscale bit levels
       grid_shape : tuple
           Number of blocks in rows and columns

       Returns
       -------
       img_blocked : array
           Array (matrix) with block types. The spatial
           order (as in the original image) is preserved
    '''
    img_blocked = np.zeros(grid_shape[0] * grid_shape[1])
    # make sure has the same columns as all other blocks
    bf = bf.reindex(columns=cols).fillna(0)
    # only if index is set (foreground blocks)
    img_blocked[bf.index] = km_block.predict(bf) + 1
    img_blocked = img_blocked.reshape(grid_shape)
    return img_blocked

def get_supblocks(img_blocked, thresh, window_shape=3):
    '''Computes superblock features for both greyscale/multichannel images
       -------------------------------------------------------------------

       Parameters
       ----------
       img_blocked : array
           Grid (matrix) with block types
           (Block types are integers in range 1 ... n_block_types)
       thresh : bool
           Indicator of whether the images were thresholded or not
       window_shape : int (optional)
           Size of a sliding window, by default 3x3 window is used

       Returns
       -------
       df : DataFrame
           DataFrame with superblocks in rows and features
           in columns. Superblock features for greyscale images
           are simply block type occurence frequencies
    '''
    supblocks = view_as_windows(img_blocked,
                                window_shape=window_shape).reshape(-1,window_shape,
                                                                   window_shape)
    mid = np.ceil(window_shape/2).astype(int) - 1
    if thresh:
        fgr_supblocks = np.stack([sb for sb in supblocks if sb[mid,mid]])
        return pd.concat([get_block_counts(sb) for sb in fgr_supblocks])
    return pd.concat([get_block_counts(sb) for sb in supblocks])

def flatten_tiles(blocks):
    return np.array([block.ravel() for block in blocks])

# for multichannel images
def threshold_multichannel(imgs, perc=25, thresh=None):
    '''Threshold color (multichannel) images
       -------------------------------------
       Sets the threshold for the images for each color 
       channel separately by taking the lower quartile (default)
       of thresholds found by Otsu method. If thresh is provided,
       then thresholded based on these pre-computed values
       
       Parameters
       ----------
       imgs : list of image arrays
       perc : float (in range 0 to 100)
       thresh : list-like (optional)

       Returns
       -------
       imgs_th : list of thresholded images
    '''
    nchan = imgs[0].shape[-1]
    # compute thresholds for each channel
    if thresh is None:
        thresh = [np.percentile([threshold_otsu(img[:,:,c]) for img in imgs], q=perc) 
                  for c in range(nchan)]
    # apply the threshold to each channel
    imgs_th = [np.stack([threshold_img(img[:,:,c], thresh[c]) 
                         for c in range(nchan)], axis=-1) for img in imgs]
    return imgs_th, thresh

class SegfreeProfiler:
    def __init__(self, **kwargs):
        '''
        Segmentation-free profiler class
        --------------------------------
        Generates segmentation-free profiles for multichannel or 3D images

        Attributes
        ----------
        tile_size : tuple
        n_block_types : int
        n_supblock_types : int
        n_subset : int
        thresh : bool
        
        colors : int
        pixel_types : int
        
        km_block : KMeans object for tiles (blocks)
        km_supblock : KMeans object for superblocks

        Methods
        -------
        fit(imgs, n_init=50, random_state=1307)
        fit_transform(imgs, n_init=50, random_state=1307)
        transform(imgs)
        
        '''
        self.tile_size = kwargs.get('tile_size', None)
        self.n_block_types = kwargs.get('n_block_types', 50)
        self.n_supblock_types = kwargs.get('n_supblock_types', 30)
        self.n_subset = kwargs.get('n_subset', 10000)
        self.thresh = True
        self.thresh_val = None
        
        self.colors = 10
        self.pixel_types = None
        
        self.km_block = None
        self.km_supblock = None
        
        self.cache = dict()

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def set_param(self, **kwargs):
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
            
    def _handle_greyscale(self, imgs):
        # threshold, convert to ubyte,
        # bin the greyscale levels and compute tile features
        if self.thresh:
            imgs = [threshold_img(img, method='otsu') for img in imgs]
        imgs = [img_as_ubyte(img) for img in imgs]
        tiled_imgs = tile_images(imgs, self.tile_size)
        # total number of image tiles
        n_tiles = len(img_tiles) * tiled_imgs[0].shape[0]
        # bin greyscale levels for each image
        blockfeats = [get_greyscale_blockfeats(t) for t in tiled_imgs]
        return blockfeats, n_tiles
    
    def _handle_multichannel(self, imgs):
        # preprocess: threshold, compress the colors,
        # and compute tile features
        if self.thresh:
            if self.thresh_val is None:
                imgs, self.thresh_val = threshold_multichannel(imgs)
            else:
                imgs, _ = threshold_multichannel(imgs, thresh=self.thresh_val)
                
        # not sure if we need this
        imgs = [img_as_ubyte(img) for img in imgs]
        tiled_imgs = tile_images(imgs, self.tile_size)
        # total number of image tiles
        n_tiles = len(img_tiles) * tiled_imgs[0].shape[0]
        # compress colors
        blockfeats = [get_color_blockfeats(t) for t in tiled_imgs]
        return blockfeats, n_tiles
        
    def _kmeans_tiles(self, blockdf, n_init, random_state, downsample):
        # downsample the image tiles
        subset = range(blockdf.shape[0])
        if downsample and self.n_subset < blockdf.shape[0]:
            np.random.seed(random_state)
            subset = np.random.choice(range(blockdf.shape[0]), size=self.n_subset)
        print("Running k-means on tiles")
        self.km_block = KMeans(n_clusters=self.n_block_types,
                               n_init=n_init,
                               random_state=random_state).fit(blockdf.iloc[subset,:])
            
    def fit(self, imgs, n_init=50,
            random_state=1307,
            downsample=True):
        if imgs[0].ndim == 2:
            print("Fitting model for greyscale images")
            blockfeats, n_tiles = self._handle_greyscale(imgs=imgs)
        if imgs[0].ndim == 3:
            print("Fitting model for multichannel images")
            blockfeats, n_tiles = self._handle_multichannel(imgs=imgs)
        blockdf = pd.concat(blockfeats).fillna(0)
        self.pixel_types = blockdf.columns.values
        
        # run kmeans on blocks
        self._kmeans_tiles(blockdf=blockdf, 
                           n_init=n_init, 
                           random_state=random_state,
                           downsample=downsample)
        
        grid_shape = tuple(int(x / y) for x,y in zip(imgs[0].shape,
                                                     self.tile_size))
        tile_labels = [get_block_types(bf,
                              km_block=self.km_block,
                              cols=self.pixel_types,
                              grid_shape=grid_shape) for bf in blockfeats]
        supblocks = [get_supblocks(bl, thresh=True) for bl in tile_labels]
        
        print("Running k-means on superblocks")
        self.km_supblock = KMeans(n_clusters=self.n_supblock_types,
                                  n_init=n_init,
                                  random_state=random_state).fit(pd.concat(supblocks).fillna(0))
        # cache blocks and supblocks
        self.cache['blocks'] = tile_labels
        self.cache['supblocks'] = supblocks
        return self
    
    def _get_mean_tile(self, blocks):
        columns = range(self.n_block_types+1)
        block_mean = pd.concat([get_block_counts(bl).reindex(columns=columns)
                                for bl in blocks]).fillna(0).reset_index(drop=True)
        block_mean.columns = ['-'.join(['block', str(col)])
                           for col in block_mean.columns.values]
        return block_mean
    
    def _get_mean_superblock(self, supblocks):
        columns = range(self.n_block_types + 1)
        # re-index the feature columns of every superblock
        # so that it matches the number of block types
        supblocks = [sbf.reindex(columns=columns).fillna(0) for sbf in supblocks]
        supblock_mean = pd.concat([get_block_counts(self.km_supblock.predict(sbf))
                       for sbf in supblocks]).reset_index(drop=True)
        supblock_mean = supblock_mean.reindex(columns=range(self.n_supblock_types)).fillna(0)
        supblock_mean.columns = ['-'.join(['superblock', str(col+1)])
           for col in supblock_mean.columns.values]
        return supblock_mean
    
    def transform(self, imgs, useCache=False):
        if useCache:
            blocks = self.cache['blocks']
            supblocks = self.cache['supblocks']
        else:
            if imgs[0].ndim == 2:
                blockfeats, n_tiles = self._handle_greyscale(imgs=imgs)
            if imgs[0].ndim == 3:
                blockfeats, n_tiles = self._handle_multichannel(imgs=imgs)
            blockdf = pd.concat(blockfeats).fillna(0)

            grid_shape = tuple(int(x / y) for x,y in zip(imgs[0].shape,
                                                         self.tile_size))
            blocks = [get_block_types(bf,
                                  km_block=self.km_block,
                                  cols=self.pixel_types,
                                  grid_shape=grid_shape) for bf in blockfeats]
            supblocks = [get_supblocks(bl, thresh=True) for bl in blocks]
        # get proportion of every tile type in each image
        block_mean = self._get_mean_tile(blocks)
        # get proprotions of superblock types in each image
        supblock_mean = self._get_mean_superblock(supblocks)
        # join block and superblock profiles
        img_prof = pd.concat([supblock_mean, block_mean], axis=1)
        return img_prof
    
    def fit_transform(self, imgs, n_init=50,
                      random_state=1307,
                      downsample=True):
        return self.fit(imgs=imgs,
                 n_init=n_init,
                 random_state=random_state,
                 downsample=downsample).transform(imgs=imgs, useCache=True)
