#!/usr/bin/env python3
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


def threshold_img(img, method='yen', binary=False):
    img_ret = np.copy(img)

    if np.sum(img_ret) < 1e-18:
        return img_ret

    if method == 'yen':
        img_ret[img_ret < skimage.filters.threshold_yen(img)] = 0
    elif method == 'otsu':
        img_ret[img_ret < skimage.filters.threshold_otsu(img)] = 0
    elif method == 'triangle':
        img_ret[img_ret < skimage.filters.threshold_triangle(img)] = 0

    if binary:
        return img_ret > 0
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


# filtering out high-frequency components by FFT based on fixed thresholds
def filter_highfreq(img, keep=0.2):
    im_fft = np.fft.fft2(img)
    im_fft2 = im_fft.copy()
    r, c = im_fft2.shape
    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[int(r*keep):int(r*(1-keep))] = 0
    # Similarly with the columns:
    im_fft2[:, int(c*keep):int(c*(1-keep))] = 0
    return np.fft.ifft2(im_fft2).real