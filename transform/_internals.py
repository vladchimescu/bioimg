#!/usr/bin/env python
"""
Internal (private) functions for the 'transform' submodule
"""
import bioformats as bf
import re
import numpy as np
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
