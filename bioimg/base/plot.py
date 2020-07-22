#!/usr/env/bin python3
"""
Functions and classes for static plots
"""
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import matplotlib.colors as mcolors

color_dict = {'red': 0, 'orange': 0.1,
              'yellow': 0.16, 'green': 0.3,
              'cyan': 0.5, 'blue': 0.6,
              'purple': 0.8, 'magenta': 0.9,
              'white': None}

def rescale_array(a):
    '''Rescale float numpy array to (0,1)-range
    '''
    if (a.dtype == np.float64) or (a.dtype == np.float32):
        return (a - a.min()) / (a.max() - a.min())
    return a

def plot_channels(images, nrow, ncol, titles=None,
                  scale_x=4, scale_y=4, cmap=None,
                  hspace=0.2, wspace=0.2, bottom=0,
                  top=0.7):
    '''Plot images as a grid of subplots
       ---------------------------------
       A list of image arrays is plotted in a matrix layout

       Parameters
       ----------
       images : list
           List of np.array (image arrays). Ararys can be
           either greyscale or color 2D images
       nrow : int
           Number of rows
       ncol : int
           Numbr of columns
       titles : list or array
           List-like, plot subtitles
       scale_x : int
           Figure width parameter: w = scale_x * ncol
       scale_y : int
           Figure height parameter: h = scale_y * nrow
       cmap : string
           Name of the matplotlib colormap. Default to viridis
       hspace : float (optional)
           proportion of height reserved for spacing between subplots
       wspace : float (optional)
           proportion of width reserved for spacing between subplots
       bottom : float (optional)
           bottom of the subplots of the figure
       top : float (optional)
           top of the subplots of the figure
    '''
    plt.figure(figsize=(scale_x * ncol, scale_y * nrow))
    plt.subplots_adjust(hspace=hspace, wspace=wspace, top=top, bottom=bottom)
    for i in range(len(images)):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(images[i], cmap=cmap)
        if titles is not None:
            plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def colorize(image, hue, saturation=1):
    """ Add color of the given hue to an RGB image.

    By default, set the saturation to 1 so that the colors pop!
    """
    if hue is None:
        return image
    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return color.hsv2rgb(hsv)


def combine_channels(images, colors, blend=None, gamma=None):
    '''Plot images as an RGB overlay
       ------------------------------
       A list of image arrays is combined into a single
       color image.

       Parameters
       ----------
       images : list
           List of np.array (image arrays). List elements are
           interpreted as individual color channels
       colors : list of strings
           List of color names: one of red, orange, yellow,
           green, cyan, blue, purple, magenta, white
       blend : list of floats (optional)
           Controls color blending in the image overlay
       gamma : list or array (optional)
           Gamma correction factor for individual images
    '''
    # rescale each channel to be in the range (0,1)
    images = [rescale_array(img) for img in images]
    if blend is None:
        blend = [0.5] * len(images)
    if gamma is not None:
        images = [img**g for img, g in zip(images, gamma)]

    
    images = [color.gray2rgb(img) for img in images]
    # color the images
    images = [colorize(img, hue=color_dict[c])
              for img, c in zip(images, colors)]
    images = [b * img for img, b in zip(images, blend)]
    # make sure that the images are in (0,1) range if dtype='float'    
    return rescale_array(sum(images))


def show_bbox(img, bbox, color='white', lw=2, size=12):
    '''Display bounding boxes of the segmentation
       ------------------------------------------
       Show the original intensity image or RGB overlay
       together with the bounding boxes of labelled regions

       Parameters
       ----------
       img : array
           Intensity or RGB image
       bbox: list / array of tuples
           Each tuple represents the bounding box image
           coordinates (xmin, xmax, ymin, ymax)
       color : string
           Color of the bounding boxes
       lw : float
           Linewidth of the bounding boxes
       size : int
           Figure size
    '''
    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    ax.imshow(img)
    for bb in bbox:
        start = (bb[0], bb[2])
        extent = (bb[1] - bb[0],
                  bb[3] - bb[2])
        rec = plt.Rectangle(xy=start,
                            width=extent[1],
                            height=extent[0], color=color,
                            linewidth=lw, fill=False)
        ax.add_patch(rec)
    ax.axis('off')


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def diverge_map(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635)):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mcolors.ColorConverter().to_rgb
    if isinstance(low, str): low = c(low)
    if isinstance(high, str): high = c(high)
    return make_colormap([low, c('white'), 0.5, c('white'), high])

