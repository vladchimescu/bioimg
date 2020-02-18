#!/usr/bin/env python
"""
Microscopy image input/output functions
Author: Vladislav Kim
"""
import os
import re
import numpy as np
import bioformats as bf
from skimage import img_as_uint

from _internals import get_img_metadata


# function for reading in .czi images
def load_imgstack(fname):
    # image metadata
    imgmd = get_img_metadata(fname)
    # image reader object
    rdr = bf.ImageReader(fname, perform_init=True)

    # initialize an empty numpy array to store the image data
    # the size of the array is
    # (Z, X, Y, C)
    imgarray = np.zeros((imgmd.Pixels.get_SizeZ(),
                         imgmd.Pixels.get_SizeX(),
                         imgmd.Pixels.get_SizeY(),
                         imgmd.Pixels.get_channel_count()))

    # read in the image into the array
    for c in range(0, imgmd.Pixels.get_channel_count()):
        for z in range(0, imgmd.Pixels.get_SizeZ()):
            imgarray[z, :, :, c] = rdr.read(c=c, z=z)
    print("Image size: %3d x %3d x %3d x %3d" % imgarray.shape)
    return imgarray


# function for reading in TIFF images
def read_tiff(fname, verbose=True):
    imgmd = get_img_metadata(fname)
    rdr = bf.ImageReader(fname, perform_init=True)

    imgarray = np.zeros((imgmd.Pixels.get_SizeX(),
                         imgmd.Pixels.get_SizeY()))
    imgarray = rdr.read()

    if verbose:
        print("Image size: %3d x %3d" % imgarray.shape)

    return imgarray


def load_image_series(path, imgfiles):
    imgmd = get_img_metadata(os.path.join(path, imgfiles[0]))

    imgarray = np.zeros((len(imgfiles),
                         imgmd.Pixels.get_SizeX(),
                         imgmd.Pixels.get_SizeY()))

    for i, fname in enumerate(imgfiles):
        imgarray[i, :, :] = read_tiff(os.path.join(path, fname),
                                      verbose=False)

    print("Series of %d images of size: %3d x %3d" % imgarray.shape)
    return imgarray


# write TIFF series (image stack, multichannel)
def write_tiff_series(img_w, pathname):
    for z in range(img_w.shape[0]):
        for c in range(img_w.shape[3]):
            bf.write_image(pathname=pathname,
                           pixels=img_as_uint(img_w[z, :, :, c]),
                           pixel_type=bf.PT_UINT16,
                           z=z, c=c,  size_c=img_w.shape[3],
                           size_z=img_w.shape[0])
