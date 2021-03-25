#!/usr/bin/env python3
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import regionprops_table
from skimage.util import img_as_ubyte
from skimage.transform import resize
import mahotas as mht
from ..base.process import threshold_img

KEYS = ['area',
        'mean_intensity',
        'centroid',
        'convex_area', 'eccentricity',
        'equivalent_diameter',
        'euler_number', 'extent',
        'orientation', 'perimeter', 'solidity',
        'filled_area',
        'major_axis_length',
        'minor_axis_length',
        'inertia_tensor', 'inertia_tensor_eigvals',
        'moments_central',
        'moments_hu',
        'moments_normalized',
        'weighted_moments_central',
        'weighted_moments_hu',
        'weighted_moments_normalized']

# create a data frame with texture features
GLCM_PROPS = ['contrast', 'dissimilarity',
              'homogeneity',
              'ASM', 'energy', 'correlation']

F_HARALICK = """AngularSecondMoment Contrast Correlation Variance
                    InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
                    DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()


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

def compute_haralick(cell, d):
    '''Compute Haralick texture features
       ---------------------------------

       Parameters
       ----------
       cell : image array
           image patch (e.g. bounding box of a cell, organoid, etc)
       d : int
           pixel distance for grey-level co-occurence matrix computation
       
       Returns
       -------
       df : DataFrame
           DataFrame with 13 Haralick texture features calculated for
           4 spatial directions (in 2D)
    '''
    names = [x + '-d' + str(d) + '-' + str(y) for y in range(4) for x in F_HARALICK]
    values = mht.features.haralick(img_as_ubyte(cell),
                                   distance=d,
                                   ignore_zeros=False).ravel()

    return pd.DataFrame({k : [v] for k,v in zip(names, values)})

def compute_zernike(cell, r=15, deg=12):
    '''Compute Zernike moments
       -----------------------

       Parameters
       ----------
       cell : image array
           image patch (e.g. bounding box of a cell, organoid, etc)
       r : int
           maximum radius, in pixels, of Zernike polynomials
       deg : int
           degree of Zernike polynomials

       Returns
       -------
       df : DataFrame
           DataFrame with Zernike moments
    '''
    values = mht.features.zernike_moments(img_as_ubyte(cell),
                                          radius=r, degree=deg)
    names = ['zernike' + '-r' + str(r) + '-' + str(i) for i in range(len(values))]
    return pd.DataFrame({k : [v] for k,v in zip(names, values)})


def compute_region_props(cell, keys=KEYS,
                         thresh='otsu',
                         texture='glcm',
                         zernike=True,
                         distances=[3, 5, 7],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         zernike_radii=[10,12],
                         zernike_deg=12):
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
       thresh : string or float
           Thresholding method to suppress the background in bounding
           boxes. If string, method name (e.g. 'otsu', 'yen', 'li', etc).
           If int/float, then the value is used to threshold the image
           with img > thresh
       texture : string
           Possible values are ['glcm', 'haralick', 'both']. If 'glcm' is
           chosen (default), texture features are computed using
           skimage.feature.greycoprops. Option 'haralick' computes
           13 Haralick features for 4 spatial directions (total 52 features)
           using mahotas.features.haralick. If 'both', then both Haralick and
           skimage texture features are computed
       zernike : bool
           Compute Zernike moments (default `True`)
       distance : list of ints
           List of pixel pair distance offsets, passed to
           skimage.feature.greycomatrix
       angles : list of floats
           List of pixel pair angles in radians, passed to
           skimage.feature.greycomatrix
       zernike_radii : array-like
           List of radii for Zernike polynomial evaluation
       zernike_deg : int
           Highest degree of Zernike polynomials to compute

       Returns
       -------
       df : DataFrame
           DataFrame with morphological properties
    '''
    bw = threshold_img(cell, method=thresh, binary=True)
    prop_df = pd.DataFrame(regionprops_table(bw.astype('int'),
                                        cell, properties=keys))    
    if texture == 'glcm' or texture == 'both':
        glcm = greycomatrix(img_as_ubyte(cell),
                        distances=distances,
                        angles=angles)
        texture_df = pd.concat([glcm_to_dataframe(glcm, prop=p)
                                for p in GLCM_PROPS], axis=1)
        prop_df = pd.concat([prop_df, texture_df], axis=1)
        
    if texture == 'haralick' or texture == 'both':
        texture_df = pd.concat([compute_haralick(cell, d=d)
                                for d in distances], axis=1)
        prop_df = pd.concat([prop_df, texture_df], axis=1)
    if zernike:
        zernike_df = pd.concat([compute_zernike(cell, r=r, deg=zernike_deg)
                                for r in zernike_radii], axis=1)
        prop_df = pd.concat([prop_df, zernike_df], axis=1)
    return prop_df


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
       params : dict (optional)
           Dictionary of parameters for morphological feature computation
           passed to compute_region_props() function. The user can modify
           the parameters before running `compute_props` method (see 'Examples')
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
       >>> # change default texture features to Haralick (Mahotas implementation)
       >>> imgx_test.params['texture'] = 'haralick'
       >>> imgx_test = imgx_test.compute_props()
       >>> # or initialize with channel names as n_chan argument
       >>> imgx_test = ImgX(img=img**0.4, bbox=bbox,
                            n_chan=['hoechst', 'ly', 'calcein'])
       >>> imgx_test = imgx_test.compute_props()
       >>> img_df = imgx_test.get_df()
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
        self.params = {'thresh' : 'otsu',
                       'texture' : 'glcm',
                       'zernike': True,
                       'distances': [3, 5, 7],
                       'angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
                       'zernike_radii': [15,18, 20],
                       'zernike_deg': 12}

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def _get_features(self, img, c=None):        
        # compute features for all the bboxes
        cellbb = [img[x[2]:x[3], x[0]:x[1]] for x in self.bbox]
        data_list = [compute_region_props(cell=cell, **self.params) for cell in cellbb]
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
            img_gray = self.img
            if self.img.ndim == 3:
                img_gray = rgb2gray(self.img)
            self._get_features(img=img_gray)
        return self

    def get_df(self):
        '''Returns DataFrame with image features
           -------------------------------------
           Run this function after `compute_props` to
           return a copy of image data as pandas.DataFrame
        '''
        if type(self.data) == dict and len(self.data) > 1:
            df = pd.concat(self.data, axis=1)
            df.columns = df.columns.droplevel()
        if type(self.data) == pd.DataFrame:
            df = self.data.copy()
        return df
