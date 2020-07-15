#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from skimage.util import view_as_blocks, view_as_windows
from skimage.util import img_as_ubyte

def get_blocktype(a, nbits):
    block_type = {i : np.sum(a == i) / a.size for i in nbits}
    return pd.DataFrame([block_type])

def get_blockfeats(img_tiles, nbits):
     return pd.concat([get_blocktype(img_tiles[i], nbits=nbits) for i in range(img_tiles.shape[0])]).reset_index(drop=True)


def get_supblocks(bf, km_block, grid_shape, window_shape=3):
    img_blocked = km_block.predict(bf).reshape(grid_shape)
    supblocks = view_as_windows(img_blocked, window_shape=window_shape).reshape(-1,window_shape,window_shape)
    # correct here: instead of range(50) -> n_supblock_types
    return pd.concat([get_blocktype(supblocks[i], nbits=range(50)) for i in range(supblocks.shape[0])])

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
        return [view_as_blocks(img,
                               block_shape=self.tile_size).reshape(-1, *self.tile_size) for img in imgs]        

    def fit(self, imgs, n_init=50, random_state=1307):
        img_tiles = self.tile_images(imgs)
        nbits = np.unique(np.stack(img_tiles))
        blockfeats = [get_blockfeats(t, nbits=nbits) for t in img_tiles]
        self.km_block = KMeans(n_clusters=self.n_block_types,
                               n_init=n_init,
                               random_state=random_state).fit(pd.concat(blockfeats))
        grid_shape = tuple(int(x / y) for x,y in zip(imgs[0].shape, self.tile_size))
        supblocks = [get_supblocks(bf,
                                   km_block=self.km_block,
                                   grid_shape=grid_shape) for bf in blockfeats]
        self.km_supblock = KMeans(n_clusters=self.n_supblock_types,
                                  n_init=n_init,
                                  random_state=random_state).fit(pd.concat(supblocks))
        return self

    def transform(self, imgs):
        img_tiles = self.tile_images(imgs)
        nbits = np.unique(np.stack(img_tiles))
        blockfeats = [get_blockfeats(t, nbits=nbits) for t in img_tiles]
        grid_shape = tuple(int(x / y) for x,y in zip(imgs[0].shape, self.tile_size))
        block_mean = pd.concat([bf.agg('mean') for bf in blockfeats], axis=1).T
        block_mean.columns = ['-'.join(['pixel', str(col)])
                           for col in block_mean.columns.values]
        supblocks = [get_supblocks(bf,
                                   km_block=self.km_block,
                                   grid_shape=grid_shape) for bf in blockfeats]
        supblock_mean = pd.concat([bf.agg('mean') for bf in supblocks], axis=1).T
        supblock_mean.columns = ['-'.join(['block', str(col)])
                           for col in supblock_mean.columns.values]
        img_profs = pd.concat([get_blocktype(self.km_supblock.predict(supblocks[i]),
                                             nbits=range(self.n_supblock_types))
                               for i in range(len(supblocks))]).reset_index(drop=True)
        img_profs.columns = ['-'.join(['superblock', str(col)])
                   for col in img_profs.columns.values]
        imgdf = pd.concat([img_profs, supblock_mean,  block_mean], axis=1)
        return imgdf
