#!/usr/bin/env python
"""
Script for generating initial segmentation maps
in leukemia entities in HPC environment
"""
import javabridge
import bioformats as bf
import skimage
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os
import sys

from skimage.morphology import disk, binary_erosion
from skimage.morphology import white_tophat
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.measure import regionprops

sys.path.append('..')
from base.utils import load_imgstack
from transform.process import threshold_img
from segment.cv_methods import filter_segm


def segment_connected_components(img):
    '''Initial segmentation for leukemia coculture
       -------------------------------------------
       Use simple connected component labelling to
       segment the nuclei of leukemia and stroma cells.
       The initial segmentation map preserves only regions
       within the specified area and size range

       Parameters
       ----------
       img : array
           Intensity image

       Returns
       -------
       segm_out : array
           Image array with labelled regions
    '''
    img_th = binary_erosion(threshold_img(img, method='otsu',
                                          binary=True), disk(5))
    segm = label(img_th, connectivity=1)
    # change the bound values if necessary
    bounds = {'area': (500, 6000), 'perimeter': (100, 1000)}
    segm1 = filter_segm(img=img, labels=segm, bounds=bounds)
    # change 'bounds' values if necessary
    big = filter_segm(img=img, labels=segm, bounds={'area': (6000, np.inf)}) +\
        filter_segm(img=img, labels=segm, bounds={
                    'perimeter': (1000, np.inf)})
    big_obj = img*np.isin(segm, np.unique(big[big != 0]))
    img_tophat = white_tophat(big_obj, disk(25))
    # bright spots from the large regions that were filtered out
    # in the previous step
    segm2 = remove_small_objects(threshold_img(img_tophat,
                                               method='yen', binary=True),
                                 min_size=500)
    # only non-background pixels
    segm1 = (segm1 != 0)
    segm_out = label(np.logical_or(segm1, segm2))
    return segm_out


if __name__ == "__main__":
    javabridge.start_vm(class_path=bf.JARS)

    # path to the image data
    path = sys.argv[1]
    # plate identifier (e.g. '180528_Plate3')
    plate = sys.argv[2]
    print "Processing plate: " + str(plate)

    # image name
    fname = sys.argv[3]

    imgstack = load_imgstack(fname=path + plate + "/" + fname)

    # remove a 'dummy' z-axis
    img = np.squeeze(imgstack)

    javabridge.kill_vm()
