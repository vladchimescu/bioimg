#!/usr/bin/env python
"""
Classical computer vision segmentation algorithms
such as watershed or spot detection
"""

import scipy.ndimage as nd
import numpy as np
import pandas as pd
from skimage.morphology import disk, binary_closing, binary_erosion
from skimage import feature
from skimage import morphology
from skimage import segmentation
from skimage.feature import shape_index, blob_log
from skimage.filters import gaussian
from skimage.measure import label
import skimage

from transform.process import adjust_contrast, elevation_map
from transform.process import threshold_stack, tophat_stack
from transform.process import threshold_img


def segment_connected_components(img):
    img_th = binary_erosion(threshold_img(img, method='otsu',
                                          binary=True), disk(5))
    segm = label(img_th, connectivity=1)
    pass


def get_feattable(feats, keys):
    '''Get region property summary as DataFrame
       ----------------------------------------
       Subsets regionprops object to selected keys and
       returns a DataFrame

       Parameters
       ----------
       feats : RegionProperties object
       keys : list of strings

       Returns
       -------
       DataFrame : pd.DataFrame of selected features

    '''
    return pd.DataFrame({key: [f[key] for f in feats] for key in keys})


def get_bounds(feat_df, key, bounds):
    '''Find regions within the RegionProperties bounded range
       ------------------------------------------------------
       Returns a boolean array for image regions whose properties
       are within the specified bounds

       Parameters
       ----------
       feat_df : DataFrame
           Region properties of the image as DataFrame
       key : key of a dict
       bounds : tuple
           Tuple of length 2 with lower_bound = bounds[0] and
           upper_bound = bounds[1]
    '''
    series = np.logical_and(feat_df[key] > bounds[0], feat_df[key] < bounds[1])
    return series.values


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


# below are functions for surface marker segmentation (need to be edited)
# now find spots for all "clusters"
def detect_spots(img, max_sigma=10, min_sigma=4, num_sigma=10, overlap=0.5):
    blobs_log = blob_log(img, max_sigma=max_sigma,
                         min_sigma=min_sigma, num_sigma=num_sigma, overlap=overlap)
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
    return blobs_log


# converts 'nan' values to -1
# (this is done since the shape index transforms the image
# to np.float array with values [-1,1])
def nantonum(img, pad=-1):
    img_r = np.copy(img)
    img_r[np.isnan(img_r)] = pad
    return img_r


# a single image
def segment_bite(img, max_sigma=10, min_sigma=4, num_sigma=10, overlap=0.5, method='yen'):
    img_adj = adjust_contrast(img)
    img_th = threshold_img(img_adj, method=method)
    img_shape = shape_index(img_th)
    spots = detect_spots(nantonum(img_shape),
                         max_sigma=max_sigma,
                         min_sigma=min_sigma,
                         num_sigma=num_sigma,
                         overlap=overlap)
    return img_shape, spots


def segment_bite_list(imglist, ch, max_sigma=10, min_sigma=4, num_sigma=10, overlap=0.5, method='yen'):
    img_adj = [adjust_contrast(i[:, :, int(ch)]) for i in imglist]
    img_th = [threshold_img(i, method=method) for i in img_adj]
    img_shape = [shape_index(i) for i in img_th]
    spots = [detect_spots(nantonum(i),
                          max_sigma=max_sigma,
                          min_sigma=min_sigma,
                          num_sigma=num_sigma,
                          overlap=overlap) for i in img_shape]
    return img_shape, spots


def get_binary_mask(img, method='yen', gamma=0.3):
    reg_th = gaussian(img**gamma)
    if method is 'yen':
        reg_th[reg_th < skimage.filters.threshold_yen(reg_th)] = 0
    elif method is 'triangle':
        reg_th[reg_th < skimage.filters.threshold_triangle(reg_th)] = 0
    elif method is 'otsu':
        reg_th[reg_th < skimage.filters.threshold_otsu(reg_th)] = 0
    elif method is 'mean':
        reg_th[reg_th < skimage.filters.threshold_mean(reg_th)] = 0
    reg_th[reg_th > 0] = 1
    return reg_th


# return the (x,y)-coordinates of the cell centroids
def get_circle_centers(spots, shape):
    indices = spots[:, :2]
    indices = indices.astype(np.int)

    centers = np.zeros(shape)
    centers[indices[:, 0], indices[:, 1]] = 1

    return centers


def segment_surface_marker(img, ch, spots, thresh='triangle',
                           disk_size=7, sigma=200, rmbg=False):
    bgsub = img[:, :, ch]
    print "Channel: " + str(ch)
    if rmbg:
        print "Background subtraction in channel " + str(ch)
        bg = gaussian(img[:, :, ch], sigma=sigma)
        bgsub = img[:, :, ch] - bg + np.mean(bg)
        # check if negative values are introduced by background subtraction
        if np.min(bgsub) < 0:
            bgsub[bgsub < 0] = np.min(bgsub[bgsub > 0])

    img_th = get_binary_mask(bgsub, method=thresh)
    img_cl = binary_closing(img_th, disk(disk_size))
    spots = np.append(spots, np.zeros((len(spots), 1)), axis=1)
    for s, sp in enumerate(spots):
        y, x, r = np.rint(sp[:3])
        # get pixels in the square mask
        sqy = np.arange(y - r, y + r)
        sqx = np.arange(x - r, x + r)

        sqy = sqy[sqy < img.shape[0]]
        sqx = sqx[sqx < img.shape[1]]

        sq = np.array(np.meshgrid(sqy, sqx), dtype=np.int).T.reshape(-1, 2)
        center = np.tile(np.array((y, x)), sq.shape[0]).reshape(-1, 2)
        circ = sq[np.where(np.sum((sq - center)**2, axis=1) <= r**2)]
        mask = np.zeros(img_cl.shape)
        mask[circ[:, 0], circ[:, 1]] = 1
        spots[s, -1] = np.sum(img_cl * mask)

    return spots
