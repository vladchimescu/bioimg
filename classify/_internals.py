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


def make_instances(mip, fname, w=40, h=40):
    calhoe = equalize_adapthist(np.dstack((mip[:, :, 1],
                                           mip[:, :, 2],
                                           mip[:, :, 0])))
    bb = []
    with open(fname) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            bb.append(row)

    bb = [np.array(b, dtype=np.int) for b in bb]
    cellbb = [calhoe[x[2]:x[3], x[0]:x[1], :] for x in bb]

    cellbb_norm = [resize(cb, (w, h), anti_aliasing=True) for cb in cellbb]

    return cellbb_norm


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


def str_gen(num):
    if num is 1:
        return ['']
    elif num > 1:
        numli = range(num)
        return [str(n) for n in numli]


def forfor(a):
    return [item for sublist in a for item in sublist]


def array_if(x):
    if type(x) is np.ndarray:
        return x
    else:
        return np.array([x])


def compute_img_features(cell, exclude, square_size=3,
                         distances=[3, 5, 7],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    thresh = threshold_yen(cell)
    bw = closing(cell > thresh, square(square_size))
    # label image regions
    labelled = label(bw)
    feats = regionprops(labelled, cell)

    areas = [f.area for f in feats]
    index, element = max(enumerate(areas), key=itemgetter(1))

    feat_dict = {}
    for f in feats[index]:
        if f not in exclude:
            feat_dict[f] = feats[index][f]

    # feature in X_train format
    X_feat = np.concatenate([array_if(i) for i in feat_dict.values()])

    # compute texture features from the greyscale co-occurence matrix
    glcm = greycomatrix(img_as_ubyte(cell),
                        distances=distances,
                        angles=angles)
    texture_feat = greycoprops(glcm)
    X_feat = np.concatenate((X_feat, texture_feat.ravel()))

    return X_feat


def get_regionprop_feats(mip_rgb, exclude):
    X_hoe = compute_img_features(cell=mip_rgb[:, :, 2],
                                 exclude=exclude)
    X_ca = compute_img_features(cell=mip_rgb[:, :, 1],
                                exclude=exclude)
    X_ly = compute_img_features(cell=mip_rgb[:, :, 0],
                                exclude=exclude)
    X_prop = np.concatenate((X_hoe, X_ca, X_ly))

    return X_prop
