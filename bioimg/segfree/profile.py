#!/usr/bin/env python3
import numpy as np
import pandas as pd
from skimage.util import view_as_blocks, view_as_windows

def get_blocktype(a, nbits):
    block_type = {i : np.sum(a == i) / a.size for i in nbits}
    return pd.DataFrame([block_type])

def get_blockfeats(img_tiles, nbits):
     return pd.concat([get_blocktype(img_tiles[i], nbits=nbits) for i in range(img_tiles.shape[0])]).reset_index(drop=True)


def get_supblocks(bf, km_block, grid_shape, window_shape=3):
    img_blocked = km_block.predict(bf).reshape(grid_shape)
    supblocks = view_as_windows(img_blocked, window_shape=window_shape).reshape(-1,window_shape,window_shape)
    return pd.concat([get_blocktype(supblocks[i], nbits=50) for i in range(supblocks.shape[0])])

class SegfreeProfiler:
    def __init__(self):
        '''
        Parameters
        ----------
        imgs : list of images
        tile_size : tuple
        km_block : KMeans object for tiles (blocks)
        km_supblock : KMeans object for superblocks
        '''
        self.imgs = None
        self.blockfeats = None
        self.tile_size = None
        self.km_block = None
        self.km_supblock = None

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def set_param(self, **kwargs):
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

    def preprocess_images(self, thresh=True):
        if thresh:
            self.imgs = [threshold_img(img, method='otsu') for img in self.imgs]
        self.imgs = [img_as_ubyte(img) for img in self.imgs]

    def tile_images(self):
        return [view_as_blocks(img, block_shape=self.tile_size) for img in self.imgs] 
       

    def fit(self, n_clusters=50, n_init=50, random_state=1307):
        img_tiles = self.tile_images()
        nbits = np.unique(np.stack(self.blocks))
        self.blockfeats = [get_blockfeats(t, nbits=nbits) for t in img_tiles]
        self.km_block = KMeans(n_clusters=n_clusters,
                               n_init=n_init,
                               random_state=random_state).fit(pd.concat(self.blockfeats))

    def transform(self, newimgs):
        pass
