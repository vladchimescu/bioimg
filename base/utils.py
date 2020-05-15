#!/usr/bin/env python
"""
Microscopy image input/output functions
Author: Vladislav Kim
"""
import os
import numpy as np
import bioformats as bf
from skimage import img_as_uint
import re
import xml.etree.cElementTree as ET


# image metadata using Bioformats OME XML parser
def get_img_metadata(fname):
    omexml = bf.get_omexml_metadata(fname)
    md = bf.OMEXML(omexml)
    imgmd = md.image()
    return imgmd

# get physical scale in Z-axis direction


def get_physicalSizeZ(fname):
    omexml = bf.get_omexml_metadata(fname)
    xml = bytes(bytearray(omexml, encoding='utf-8'))
    root = ET.fromstring(xml)

    for child in root.iter():
        if re.search("Pixels", child.tag) is not None:
            scaleZ = child.attrib['PhysicalSizeZ']
    return np.float64(scaleZ)

# return physical scale in all directions (ZXY)


def get_physical_scale(fname):
    imgmd = get_img_metadata(fname)
    physicalSizeZ = get_physicalSizeZ(fname)

    return [physicalSizeZ, imgmd.Pixels.get_PhysicalSizeX(),
            imgmd.Pixels.get_PhysicalSizeY()]


# function for reading in .czi images
def load_imgstack(fname, verbose=False):
    # image metadata
    imgmd = get_img_metadata(fname)
    # image reader object
    rdr = bf.ImageReader(fname, perform_init=True)

    # initialize an empty numpy array to store the image data
    # the size of the array is
    # (Z, X, Y, C)
    n_stacks = imgmd.Pixels.get_SizeZ()
    n_col = imgmd.Pixels.get_channel_count()

    if n_stacks > 1 or n_col > 1:
        imgarray = np.zeros((n_stacks,
                             imgmd.Pixels.get_SizeX(),
                             imgmd.Pixels.get_SizeY(),
                             n_col))

        # read in the image into the array
        for c in range(0, imgmd.Pixels.get_channel_count()):
            for z in range(0, imgmd.Pixels.get_SizeZ()):
                imgarray[z, :, :, c] = rdr.read(c=c, z=z)
    else:
        imgarray = rdr.read()

    if verbose:
        print("Image size: %3d x %3d x %3d x %3d" % imgarray.shape)
    return imgarray


# function for reading in TIFF images
def read_image(fname, verbose=True):
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
        imgarray[i, :, :] = read_image(os.path.join(path, fname),
                                       verbose=False)

    print("Series of %d images of size: %3d x %3d" % imgarray.shape)
    return imgarray


def write_image(img, path, z=0, c=0, size_c=1, size_z=1,
                channel_names=None):
    bf.write_image(pathname=path,
                   pixels=img_as_uint(img),
                   pixel_type=bf.PT_UINT16,
                   z=z, c=c, size_c=size_c,
                   size_z=size_z,
                   channel_names=channel_names)


def write_imgstack(img, path, size_z, size_c):
    if size_z == 1:
        print("Adding an extra dimension for z")
        img = img[None, ...]
    if size_c == 1:
        print("Adding an extra dimension for c")
        img = img[..., None]

    if img.shape[0] != size_z | img.shape[-1] != size_c:
        raise ValueError(
            "Channel and z-stack order are mixed up")

    for z in range(size_z):
        for c in range(size_c):
            write_image(img[z, :, :, c], path=path,
                        z=z, c=c, size_z=size_z,
                        size_c=size_c)
