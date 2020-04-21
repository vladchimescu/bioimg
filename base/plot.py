#!/usr/env/bin python
"""
Functions and classes for static plots
"""
import matplotlib.pyplot as plt
from skimage import color

color_dict = {'red': 0, 'orange': 0.1,
              'yellow': 0.16, 'green': 0.3,
              'cyan': 0.5, 'blue': 0.6,
              'purple': 0.8, 'magenta': 0.9,
              'white': None}


def plot_gallery(images, titles, h, w, c, n_row=3, n_col=4):
    """Helper function to plot a gallery of image instances"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w, c)))
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def plot_channels(images, nrow, ncol, titles=None,
                  scale_x=4, scale_y=4, cmap=None):
    plt.figure(figsize=(scale_x * ncol, scale_y * nrow))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
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
    if blend is None:
        blend = [0.5] * len(images)
    if gamma is not None:
        images = [img**g for img, g in zip(images, gamma)]

    images = [color.gray2rgb(img) for img in images]
    # color the images
    images = [colorize(img, hue=color_dict[c])
              for img, c in zip(images, colors)]
    images = [b * img for img, b in zip(images, blend)]
    return sum(images)


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
