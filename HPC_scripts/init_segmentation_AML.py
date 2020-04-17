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

from ..base.utils import load_imgstack


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
