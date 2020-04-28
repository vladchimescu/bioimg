#!/usr/bin/env python
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from skimage.util import img_as_ubyte
from ..transform.process import threshold_img
from ..base.future_versions import regionprops_table

KEYS = ['area', 'centroid',
        'convex_area', 'eccentricity',
        'equivalent_diameter',
        'euler_number', 'filled_area',
        'major_axis_length',
        'minor_axis_length',
        'moments', 'moments_central',
        'moments_hu', 'moments_normalized',
        'orientation', 'perimeter', 'solidity']

# create a data frame with texture features
GLCM_PROPS = ['contrast', 'dissimilarity',
              'ASM', 'energy', 'correlation']


def glcm_to_dataframe(glcm, prop):
    '''Compute GLCM property
    '''
    mat = greycoprops(glcm, prop=prop)
    columns = ['-'.join([prop, str(i)]) for i in range(len(mat.ravel()))]
    return pd.DataFrame(mat.ravel().reshape(1, -1),
                        columns=columns)


def compute_region_props(cell, keys=KEYS,
                         distances=[3, 5, 7],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    bw = threshold_img(cell, method='otsu', binary=True)
    df = pd.DataFrame(regionprops_table(bw.astype('int'),
                                        cell, properties=keys))

    glcm = greycomatrix(img_as_ubyte(cell),
                        distances=distances,
                        angles=angles)
    texture_df = pd.concat([glcm_to_dataframe(glcm, prop=p)
                            for p in GLCM_PROPS], axis=1)

    return pd.concat([df, texture_df], axis=1)

# old version of the function


class ImgX:
    '''Labelled image class

       Examples
       --------
       >>> imgx_test = ImgX(img=img**0.4, bbox=bbox)
       >>> imgx_test = imgx_test.compute_props(n_chan=3)
       >>> imgx_test = imgx_test.compute_props(n_chan=['hoechst', 'ly', 'calcein'])
    '''

    def __init__(self, img, bbox, y=None):
        self.img = img
        self.bbox = bbox
        self.y = y

        self.data = dict()
        self.target_names = None

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def _get_features(self, img, c=None):
        # compute features for all the bboxes
        cellbb = [img[x[2]:x[3], x[0]:x[1]] for x in self.bbox]
        data_list = [compute_region_props(cell=cell) for cell in cellbb]
        # all region and GLCM properties for each 'cellbb'
        prop_df = pd.concat(data_list)
        if c is not None:
            self.data[c] = prop_df
        else:
            self.data = prop_df

    def compute_props(self, n_chan=None, split=True):
        # split=True means that the color channels will be split and the
        # properties will be computed for each channel separately
        if isinstance(n_chan, int) and split:
            for c in range(n_chan):
                self._get_features(img=self.img[:, :, c], c=c)
        if hasattr(n_chan, "__len__") and split:
            for c, col in enumerate(n_chan):
                self._get_features(img=self.img[:, :, c], c=col)
        if n_chan is None or split is False:
            img_gray = rgb2gray(self.img)
            self._get_features(img=img_gray)
        return self
