#!/usr/bin/env python3
"""
Image processing functions such as
brightness, contrast adjustment, Gaussian blur, etc

Author: Vladislav Kim
"""
import scipy.ndimage as nd
import numpy as np
import skimage
from skimage import exposure
from skimage import morphology
from skimage.morphology import disk


def threshold_img(img, method='yen', binary=False, minsum=1e-10):
    '''Apply thresholding to a greyscale image
       ---------------------------------------
       Threshold an image and return either a binary image
       or the transformed image with only foreground pixels with
       non-zero values

       Parameters
       ----------
       img : np.array
           Input image, greyscale 2D
       method : string, int or float
           If string, one of 'yen', 'otsu', 'triangle'. See the documentation
           (https://scikit-image.org/docs/stable/api/skimage.filters.html). If a number is provided, then used as the intensity cutoff.
       binary : bool
           Return a binary image (0: background, 1: foreground).
           Otherwise the image is multiplied elementwise with the
           binary mask (background pixels are set to zero).

       Returns
       -------
       img_ret : np.array
           An image with background pixels set to zero and
           only foreground pixels with non-zero values
    '''
    img_ret = np.copy(img)

    if np.sum(img_ret) < minsum:
        return img_ret

    if method == 'yen':
        img_ret[img_ret < skimage.filters.threshold_yen(img)] = 0
    elif method == 'otsu':
        img_ret[img_ret < skimage.filters.threshold_otsu(img)] = 0
    elif method == 'triangle':
        img_ret[img_ret < skimage.filters.threshold_triangle(img)] = 0
    elif isinstance(method, int) or isinstance(method, float):
        img_ret[img_ret < method] = 0

    if binary:
        return img_ret > 0
    return img_ret


def adjust_contrast(img_stack, medf_size=7):
    '''Adjust contrast of an image stack using CLAHE
       ---------------------------------------------
       Parameters
       ----------
       img_stack : np.array
           Image stack. The assumed order is (z,x,y)
       medf_size : int
           Size of window for median filter
       
       Returns
       -------
       img_adj : np.array
           Contrast-adjusted image stack
    '''
    img_stack_adj = np.zeros_like(img_stack)
    for z in range(img_stack.shape[0]):
        if np.any(img_stack[z]):
            img_stack_adj[z] = nd.filters.median_filter(
                exposure.equalize_adapthist(img_stack[z]), size=medf_size)
    return img_stack_adj

def smooth_img(img_stack, medf_size=7):
    '''Apply median filter to smooth an image stack
       --------------------------------------------
       Parameters
       ----------
       img_stack : np.array
           Image stack. The assumed order is (z,x,y)
       medf_size : int
           Size of window for median filter
       
       Returns
       -------
       img_adj : np.array
           Median-filtered image stack
    '''
    img_smooth = np.zeros_like(img_stack)
    for z in range(img_stack.shape[0]):
        img_smooth[z] = nd.filters.median_filter(img_stack[z], size=medf_size)
    return img_smooth


def tophat_stack(imgstack, size=5):
    '''Apply white tophat filter to an image stack
       -------------------------------------------
       Parameters
       ----------
       img_stack : np.array
           Image stack. The assumed order is (z,x,y)
       size : int
           (Morphological) disk size
       
       Returns
       -------
       img_f : np.array
           White-tophat filtered image stack
    '''
    img_f = np.zeros_like(imgstack)
    for z in range(imgstack.shape[0]):
        img_f[z] = morphology.white_tophat(imgstack[z], disk(size))
    return img_f


def filter_highfreq(img, keep=0.2):
    '''Filter out high-frequency components by FFT
       -------------------------------------------
       Basic filtering in Fourier domain
       
       Parameters
       ----------
       img : np.array
           Input image (greyscale 2D)
       keep : float
           A float in range (0,1), specifying the
           fraction of low-frequency components
           to keep

       Returns
       -------
       img_f : np.array
           Image filtered in Fourier domain
    '''
    im_fft = np.fft.fft2(img)
    im_fft2 = im_fft.copy()
    r, c = im_fft2.shape
    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[int(r*keep):int(r*(1-keep))] = 0
    # Similarly with the columns:
    im_fft2[:, int(c*keep):int(c*(1-keep))] = 0
    return np.fft.ifft2(im_fft2).real
