#!/usr/bin/env python3
"""
Internal functions for classifiers
"""
import numpy as np
import csv
from skimage.exposure import equalize_adapthist
from skimage.transform import resize
from skimage.feature import greycomatrix, greycoprops
from skimage.util import img_as_ubyte
from skimage.filters import threshold_yen
from operator import itemgetter
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

def circleIntersection(r1, r2, d):
    # http://mathworld.wolfram.com/Circle-CircleIntersection.html
    from numpy import arccos, sqrt

    return (r1 ** 2 * arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
            + r2 ** 2 * arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
            - sqrt((-d + r1 + r2) * (d + r1 - r2)
                   * (d - r1 + r2) * (d + r1 + r2)) / 2)


def merge_spots(small_spots, big_spots):
    ''' nonconcentric = np.all(np.isin(small_spots[:,:2],
               big_spots[:,:2], invert=True), axis=1)'''
    spots = np.vstack((small_spots, big_spots))
    radii = spots[:, 2]
    distances = np.linalg.norm(spots[None, :, :2] - spots[:, None, :2], axis=2)
    intersections = circleIntersection(r1=radii, r2=radii.T,
                                       d=distances)
    intersections[(distances == 0) & ~np.eye(
        distances.shape[0], dtype=np.bool)] = 3
    merge_circles = np.where((intersections > 1) & (
        radii[:, None] > radii[None, :]))

    merge1, c1 = np.unique(merge_circles[0], return_counts=True)
    delete = np.ones(spots.shape[0], dtype=np.bool)
    delete[merge_circles[1]
           [np.isin(merge_circles[0], merge1[c1 == 1])]] = False
    # delete larger circles for 3-way intersections (or higher-order)
    delete[merge_circles[0][np.isin(merge_circles[0], merge1[c1 > 1])]] = False
    return spots[delete]


def write_boxes(miplist, spots, fout, pad=10):
    for which_ind in range(len(miplist)):
        bb = []
        for blob in spots[which_ind]:
            cy, cx, r = blob
            bb.append(np.array((max(0, cx - np.ceil(r) - pad),
                                min(cx + np.ceil(r) + pad,
                                    miplist[0].shape[0] - 1),
                                max(0, cy - np.ceil(r) - pad),
                                min(cy + np.ceil(r) + pad,
                                    miplist[0].shape[0] - 1)),
                               dtype=np.int))

        with open(fout, 'w') as f1:
            writer = csv.writer(f1, delimiter='\t', lineterminator='\n',)
            for b in bb:
                writer.writerow(b)

