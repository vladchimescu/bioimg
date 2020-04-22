#!/usr/bin/env python
import numpy as np


def make_bbox(feats, rmax, cmax, pad):
    bbox = []
    for i in range(len(feats)):
        if type(feats) == np.ndarray:
            ymin, xmin, ymax, xmax = feats[i]
        elif type(feats) == list:
            ymin, xmin, ymax, xmax = feats[i].bbox
        bb = np.array((max(0, xmin - pad),
                       min(xmax + pad, rmax - 1),
                       max(0, ymin - pad),
                       min(ymax + pad, cmax - 1)))
        bbox.append(bb)
    return bbox


def read_bbox(df, rmax, cmax, pad=20):
    bbox_array = df[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].values

    bbox = make_bbox(feats=bbox_array,
                     rmax=rmax, cmax=cmax,
                     pad=pad)
    return bbox
