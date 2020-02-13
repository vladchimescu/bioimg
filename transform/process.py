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


def threshold_stack(imgstack):
    img_ret = np.copy(imgstack)
    for z in range(imgstack.shape[0]):
        img_ret[z][img_ret[z] < skimage.filters.threshold_yen(imgstack[z])] = 0
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
