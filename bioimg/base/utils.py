#!/usr/bin/env python3
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


def get_img_metadata(fname):
    '''Read image metadata
      --------------------
      Read microscopy image metadata using
      Bioformats OME XML parser

      Parameters
      ----------
      fname : string
          Image file name

      Returns
      -------
      imgmd : Bioformats OMEXML.Image
          Image metadata as omexml.OMEXML.Image. For details
          see python-bioformats documentation
          (https://pythonhosted.org/python-bioformats/)
    '''
    omexml = bf.get_omexml_metadata(fname)
    md = bf.OMEXML(omexml)
    imgmd = md.image()
    return imgmd


def get_physicalSizeZ(fname):
    '''Get physical scale of Z-stack
       -----------------------------
       Parameters
       ----------
       fname : string
          Image file name
       Returns
       -------
       scaleZ : float
          Spacing between optical sections
    '''
    omexml = bf.get_omexml_metadata(fname)
    xml = bytes(bytearray(omexml, encoding='utf-8'))
    root = ET.fromstring(xml)

    for child in root.iter():
        if re.search("Pixels", child.tag) is not None:
            scaleZ = child.attrib['PhysicalSizeZ']
    return np.float64(scaleZ)


def get_physical_scale(fname):
    '''Get physical scale of XYZ dimensions
       ------------------------------------
       Parameters
       ----------
       fname : string
          Image file name
       Returns
       -------
       scale : tuple
           (Z, X, Y) tuple holding spacing
            between pixels in (z,x,y)
    '''
    imgmd = get_img_metadata(fname)
    physicalSizeZ = get_physicalSizeZ(fname)

    return (physicalSizeZ, imgmd.Pixels.get_PhysicalSizeX(),
            imgmd.Pixels.get_PhysicalSizeY())


def read_image(fname, verbose=True):
    '''Read a 2D greyscale image
       -----------------------
       Wrapper of bioformats.ImageReader.read()
       for a simple greyscale 2D-image in one of the
       OME-supported formats 
       (https://docs.openmicroscopy.org/bio-formats/6.5.0/supported-formats.html)
       
       Parameters
       ----------
       fname : string
          Image file name
       verbose : bool
          If `True` print image shape
       Returns
       -------
       imgarray : np.array
    '''
    imgmd = get_img_metadata(fname)
    rdr = bf.ImageReader(fname, perform_init=True)

    imgarray = np.zeros((imgmd.Pixels.get_SizeX(),
                         imgmd.Pixels.get_SizeY()))
    imgarray = rdr.read()

    if verbose:
        print("Image size: %3d x %3d" % imgarray.shape)

    return imgarray


def load_imgstack(fname, verbose=False):
    '''Load a multi-dimensional color image
       ------------------------------------
       Wrapper of bioformats.ImageReader.read()
       for multidimensional (XYZC) color images
       
       Parameters
       ----------
       fname : string
          Image file name
       verbose : bool
          If `True` print image shape
       Returns
       -------
       imgarray : np.array
          Image array in (ZXYC) order
    '''
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


def load_image_series(path, imgfiles):
    '''
    Read a series of greyscale images
    ---------------------------------
    Read a series of greyscale images (e.g. TIFF series)
    into a numpy array

    Parameters
       ----------
       fname : string
          Image file name
       verbose : bool
          If `True` print image shape
       Returns
       -------
       imgarray : np.array
          Image array in (ZXYC) order
    '''
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


def load_image_series(path, imgfiles, verbose = False):
    '''
    Read a series of greyscale images
    ---------------------------------
    Read a series of greyscale images (e.g. TIFF series)
    into a single numpy array. Supports all OME formats

    Parameters
    ----------
    path : string
        Path with the image files
    imgfiles : list
        List of image series filenames
    verbose : bool
          If `True` print image shape
    Returns
    -------
    imgarray : np.array
        Image array of shape (N, X, Y) with
        N - number of images in the series
    '''
    imgmd = get_img_metadata(os.path.join(path, imgfiles[0]))

    imgarray = np.zeros((len(imgfiles),
                         imgmd.Pixels.get_SizeX(),
                         imgmd.Pixels.get_SizeY()))

    for i, fname in enumerate(imgfiles):
        imgarray[i, :, :] = read_image(os.path.join(path, fname),
                                       verbose=False)

    if verbose:
        print("Series of %d images of size: %3d x %3d" % imgarray.shape)
    return imgarray


def write_image(img, path, z=0, c=0, size_c=1, size_z=1,
                channel_names=None):
    '''Write image to file
       -----------------------------------
       Save an image array in one of OME-supported formats.
       By default a greyscale 2D image is expected. 
      
       Parameters
       ----------
       img : np.array
           Image array to write
       path : string
           Path where to save the image
       z : int
           Indicate which optical section (z-axis)
           to write to
       c : int
           Indicate which color channel to write to
       size_c : int
           Total number of color channels
       size_z : int
           Total number of optical sections (z-stack)
    '''
    bf.write_image(pathname=path,
                   pixels=img_as_uint(img),
                   pixel_type=bf.PT_UINT16,
                   z=z, c=c, size_c=size_c,
                   size_z=size_z,
                   channel_names=channel_names)


def write_imgstack(img, path, size_z, size_c):
    '''Write a multidimensional image (image stack)
       -----------------------------------
       Save an image array in one of OME-supported formats.
       The size of z-stack and number of color channels has
       to be specified.
       For a multidimensional array (ZXYC) order is assumed.
       If size_z=1 or size_c=1, the corresponding dimension is
       padded, i.e. an empty axis is created but not output.
      
       Parameters
       ----------
       img : np.array
           Image array to write
       path : string
           Path where to save the image
       size_c : int
           Total number of color channels
       size_z : int
           Total number of optical sections (z-stack)
    '''
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
