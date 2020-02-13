#!/usr/bin/env python
"""
Classical computer vision segmentation algorithms
such as watershed or spot detection
"""

import scipy.ndimage as nd
import numpy as np
from skimage.morphology import disk
from skimage import feature
from skimage import morphology
from skimage import segmentation
import skimage

from transform.process import adjust_contrast, elevation_map
from transform.process import threshold_stack, tophat_stack


def find_markers(imgstack, perc=95, mdist=3):
    markers = np.zeros_like(imgstack)

    for i in range(imgstack.shape[0]):
        th = np.max((np.percentile(imgstack[i, :, :], perc),
                     np.percentile(imgstack, perc)))
        markers[i, :, :] = feature.peak_local_max(imgstack[i, :, :],
                                                  threshold_abs=th,
                                                  indices=False,
                                                  min_distance=mdist)
    return markers


def intersect_nd(arrA, arrB):
    return not set(map(tuple, arrA)).isdisjoint(map(tuple, arrB))


def create_mask3d(markers, disk_size=7):
    mask3d = np.zeros_like(markers)

    for z in range(markers.shape[0]):
        mask3d[z] = skimage.morphology.binary_dilation(
            markers[z], disk(disk_size))

    return mask3d

# segment based on random walker segmentation
# at the moment beta = 1e5


def segment_rw(img, mask3d, markers, spacing):
    lbl_seeds = np.zeros_like(img)
    for z in range(img.shape[0]):
        lbl_seeds[z] = nd.label(markers[z])[0]
        lbl_seeds[z][np.where(mask3d[z] == 0)] = -1
        lbl_seeds[z] = morphology.dilation(lbl_seeds[z], morphology.disk(3))

    rw_labels = segmentation.random_walker(img,
                                           lbl_seeds,
                                           beta=1e5,
                                           spacing=spacing)

    return rw_labels


# watershed segmentation (simple watershed as opposed to
# joint segmetnation of several watersheds)
def segment_watershed(img, mask3d, markers, sigma=1):
    LoG = elevation_map(img, sigma=sigma)
    return morphology.watershed(LoG, nd.label(markers)[0], mask=mask3d)


def segment_coculture(imglist, ch=0, disk_size=7):
    img_adj = [adjust_contrast(i[:, :, :, int(ch)]) for i in imglist]

    img_filt = [tophat_stack(imgstack=i, size=disk_size) for i in img_adj]

    img_th = [threshold_stack(i) for i in img_filt]

    markers = [find_markers(i) for i in img_th]

    mask3d_list = [create_mask3d(x, disk_size=disk_size)
                   for x in markers]

    wshed_labels = [segment_watershed(img=x[0], mask3d=x[1], markers=x[2])
                    for x in zip(img_adj, mask3d_list, markers)]

    return img_adj, wshed_labels

# function for coculture segmentation specific
# to certain Zeiss image data (june 2017, e.g.)


def segment_clusters(imglist):
    img_adj = [adjust_contrast(i) for i in imglist]

    markers = [find_markers(i) for i in img_adj]

    mask3d_list = [create_mask3d(x, disk_size=7)
                   for x in markers]

    wshed_labels = [segment_watershed(img=x[0], mask3d=x[1], markers=x[2])
                    for x in zip(img_adj, mask3d_list, markers)]

    return img_adj, wshed_labels
