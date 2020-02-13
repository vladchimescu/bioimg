#!/usr/bin/env python
"""
Image processing functions such as
brightness, contrast adjustment, Gaussian blur, etc

Author: Vladislav Kim

TO-DO: add gamma correction function (for an image stack)
"""
import scipy.ndimage as nd
import numpy as np
import skimage
from skimage import exposure
from skimage import morphology
from skimage.morphology import disk


# edit this function so that it can adjust contrast
# either in a single image or in an image stack
def adjust_contrast(img_stack, medf_size=7):
    img_stack_adj = np.zeros_like(img_stack)
    for z in range(img_stack.shape[0]):
        if np.any(img_stack[z]):
            img_stack_adj[z] = nd.filters.median_filter(
                exposure.equalize_adapthist(img_stack[z]), size=medf_size)
    return img_stack_adj


def tophat_stack(imgstack, size=5):
    img_f = np.zeros_like(imgstack)
    for z in range(imgstack.shape[0]):
        img_f[z] = morphology.white_tophat(imgstack[z], disk(size))
    return img_f

# edit and merge these two functions so that
# thresholding can work on either a single optical section
# or an image stack


def threshold_stack(imgstack):
    img_ret = np.copy(imgstack)
    for z in range(imgstack.shape[0]):
        img_ret[z][img_ret[z] < skimage.filters.threshold_yen(imgstack[z])] = 0
    return img_ret


def threshold_img(img, method='yen'):
    img_ret = np.copy(img)

    if np.sum(img_ret) < 1e-18:
        return img_ret

    if method is 'yen':
        img_ret[img_ret < skimage.filters.threshold_yen(img)] = 0
    elif method is 'otsu':
        img_ret[img_ret < skimage.filters.threshold_otsu(img)] = 0
    elif method is 'triangle':
        img_ret[img_ret < skimage.filters.threshold_triangle(img)] = 0
    return img_ret


def smooth_img(img_stack, medf_size=7):
    img_smooth = np.zeros_like(img_stack)
    for z in range(img_stack.shape[0]):
        img_smooth[z] = nd.filters.median_filter(img_stack[z], size=medf_size)
    return img_smooth


def elevation_map(img, sigma=1):
    LoG_calc = np.zeros_like(img)
    for z in range(img.shape[0]):
        LoG_calc[z] = nd.gaussian_laplace(img[z], sigma=sigma)
    return LoG_calc
