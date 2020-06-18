#!/usr/bin/env python3
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from skimage.util import img_as_ubyte
from ..base.process import threshold_img
from ..base.future_versions import regionprops_table

KEYS = ['area',
        'mean_intensity',
        'centroid',
        'convex_area', 'eccentricity',
        'equivalent_diameter',
        'euler_number', 'filled_area',
        'major_axis_length',
        'minor_axis_length',
        'moments', 'moments_central',
        'moments_hu',
        'orientation', 'perimeter', 'solidity']

# create a data frame with texture features
GLCM_PROPS = ['contrast', 'dissimilarity',
              'ASM', 'energy', 'correlation']


def glcm_to_dataframe(glcm, prop):
    '''Compute GLCM properties and return a DataFrame
       ----------------------------------------------
       Convert greycoprops matrix into a pandas.DataFrame

       Parameters
       ----------
       glcm : 2D array
           Matrix with greyscale co-occurence matrix (GLCM) results
       prop : string
           GLCM feature to compute: one of contrast, dissimilarity,
           ASM, energy, correlation

       Returns
       -------
       df : DataFrame
           DataFrame with GLCM feature specified by prop argument
    '''
    mat = greycoprops(glcm, prop=prop)
    columns = ['-'.join([prop, str(i)]) for i in range(len(mat.ravel()))]
    return pd.DataFrame(mat.ravel().reshape(1, -1),
                        columns=columns)


def compute_region_props(cell, keys=KEYS,
                         distances=[3, 5, 7],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    '''Compute region properties
       -------------------------
       Compute morphological properties for the provided region
       
       Parameters
       ----------
       cell : np.array
           Image array that defines the region for which the
           morphological features will be computed
       keys : list of strings
           Morphological properties to compute (e.g. area,
           mean_intensity, eccentricity, image moments, etc).
           See the documentation of skimage.measure
           (https://scikit-image.org/docs/dev/api/skimage.measure.html)
       distance : list of ints
           List of pixel pair distance offsets, passed to
           skimage.feature.greycomatrix
       angles : list of floats
           List of pixel pair angles in radians, passed to
           skimage.feature.greycomatrix

       Returns
       -------
       df : DataFrame
           DataFrame with morphological properties
    '''
    bw = threshold_img(cell, method='otsu', binary=True)
    df = pd.DataFrame(regionprops_table(bw.astype('int'),
                                        cell, properties=keys))

    glcm = greycomatrix(img_as_ubyte(cell),
                        distances=distances,
                        angles=angles)
    texture_df = pd.concat([glcm_to_dataframe(glcm, prop=p)
                            for p in GLCM_PROPS], axis=1)

    return pd.concat([df, texture_df], axis=1)


class ImgX:
    '''Labelled image class
       --------------------
       Stores a labelled (segmented) image and morphological
       properties of segmented regions (e.g. cells)

       Attributes
       ----------
       img : array
           Input array can be a greyscale or color 2D image. 
           The assumed order is (x,y,c), i.e. color axis is last.
       bbox : list of arrays
           Bounding boxes of individual segmented regions (e.g. cells)
       n_chan : int or array-like
           Number of color channels or list of color channel names. 
           Default: greyscale (`n_chan=None`)
       y : array-like (optional)
           Labels of bounding boxes (e.g. could be cell types)
       data : dict or DataFrame
           Morphological data with regions (e.g. cells) in rows and
           features in columns. If channel names are provided, these are
           prepended to column names, else columns are prefixed with 'ch-x' 
           where 'x' is the order of the channel in the image array
       target_names : array-like (optional)
           Class names of provided labels (`y`).
           If `y` is provided as array of integers (e.g. 0 - cell type A and
           (1 - cell type B) then target_names = ['cell A', 'cell B']

       Methods
       -------
       compute_props(split=True)
           Computes morphological properties for each segmented
           region. If `split=True` and the input image is multichannel
           then features are computed for each color channel separately,
           (i.e. the channels are split)

       Examples
       --------
       >>> imgx_test = ImgX(img=img**0.4, bbox=bbox, n_chan=3)
       >>> imgx_test = imgx_test.compute_props()
       >>> # or initialize with channel names as n_chan argument
       >>> imgx_test = ImgX(img=img**0.4, bbox=bbox,
                            n_chan=['hoechst', 'ly', 'calcein'])
       >>> imgx_test = imgx_test.compute_props()
    '''

    def __init__(self, img, bbox, n_chan=None, y=None):
        '''
        Parameters
        ----------
        img : array
           Input array can be a greyscale or color 2D image. 
           The assumed order is (x,y,c), i.e. color axis is last.
        bbox : list of arrays
           Bounding boxes of individual segmented regions (e.g. cells)
        n_chan : int or array-like
           Number of color channels or list of color channel names. 
           Default: greyscale (`n_chan=None`)
        y : array-like (optional)
           Labels of bounding boxes (e.g. could be cell types)
        data : dict or DataFrame
           Morphological data with regions (e.g. cells) in rows and
           features in columns. If channel names are provided, these are
           prepended to column names, else columns are prefixed with 'ch-x' 
           where 'x' is the order of the channel in the image array
        target_names : array-like (optional)
           Class names of provided labels (`y`).
           If `y` is provided as array of integers (e.g. 0 - cell type A and
           (1 - cell type B) then target_names = ['cell A', 'cell B']
        '''
        self.img = img
        self.bbox = bbox
        self.n_chan = n_chan
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
            prop_df.columns = ['-'.join(['ch', str(c), col])
                               for col in prop_df.columns.values]
            self.data[c] = prop_df
        else:
            self.data = prop_df

    def compute_props(self, split=True):
        '''Compute morphological properties of a lablled image
           ---------------------------------------------------
           Generate DataFrame of morphological properties of
           the labelled regions (e.g. cells). If input image is
           multichannel, then features are computed separately
           for each color channel by default (`split=True`)

           Parameters
           ----------
           split : bool
               If `split=True` (default) and an image has multiple
               color channels, then morphological properties are 
               computed separately for each channel and stored in
               columns prefixed by channel name (if provided, e.g. Hoechst-area) or
               the order of the channel in the array (e.g. 
               ch-0-area). If `split=False` features are computed for
               a greyscale image
        '''
        # split=True means that the color channels will be split and the
        # properties will be computed for each channel separately
        if isinstance(self.n_chan, int) and split:
            for c in range(self.n_chan):
                self._get_features(img=self.img[:, :, c], c=c)
        if hasattr(self.n_chan, "__len__") and split:
            for c, col in enumerate(self.n_chan):
                self._get_features(img=self.img[:, :, c], c=col)
        if self.n_chan is None or split is False:
            img_gray = rgb2gray(self.img)
            self._get_features(img=img_gray)

        if len(self.data) > 1:
            self.data = pd.concat(self.data, axis=1)
            self.data.columns = self.data.columns.droplevel()

        return self
