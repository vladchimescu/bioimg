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
def read_tiff(fname):
    imgmd = get_img_metadata(fname)
    rdr = bf.ImageReader(fname, perform_init=True)

    imgarray = np.zeros((imgmd.Pixels.get_SizeX(),
                         imgmd.Pixels.get_SizeY()))
    imgarray = rdr.read()

    print("Image size: %3d x %3d" % imgarray.shape)

    return imgarray


def load_image_series(path, well):
    tif_files = [f for f in os.listdir(path)
                 if re.match('.*\.tiff', f)]
    p = "^" + well + ".+$"
    pos_list = [re.search(p, f).group() for f in tif_files
                if re.search(well, f) is not None]

    imgmd = get_img_metadata(path + tif_files[0])

    # get the num of all the fields of view, z-stacks and color channels
    n_fields = len(set([re.search('f\d+', f).group() for f in pos_list
                        if re.search(well, f) is not None]))
    n_stacks = len(set([re.search('p\d+', f).group() for f in pos_list
                        if re.search(well, f) is not None]))
    n_channels = len(set([re.search('ch\d+', f).group() for f in pos_list
                          if re.search(well, f) is not None]))

    img_list = []
    for well_pos in range(n_fields):
        imgarray = np.zeros((n_stacks,
                             imgmd.Pixels.get_SizeX(),
                             imgmd.Pixels.get_SizeY(),
                             n_channels))
        for z in range(n_stacks):
            for c in range(n_channels):
                z_pat = "p"+str(z+1).zfill(2) + "-ch"
                # pattern for slice z and color channel c
                p = "r\d+c\d+f" + str(well_pos+1).zfill(2) + \
                    z_pat + str(c+1) + ".+$"
                fname = [re.search(p, i).group()
                         for i in pos_list
                         if re.search(p, i) is not None]
                if len(fname) is not 0:
                    imgarray[z, :, :, c] = read_tiff(path + fname[0])

        img_list.append(imgarray)

    return img_list


# write TIFF series (image stack, multichannel)
def write_tiff_series(img_w, pathname):
    for z in range(img_w.shape[0]):
        for c in range(img_w.shape[3]):
            bf.write_image(pathname=pathname,
                           pixels=img_as_uint(img_w[z, :, :, c]),
                           pixel_type=bf.PT_UINT16,
                           z=z, c=c,  size_c=img_w.shape[3],
                           size_z=img_w.shape[0])
