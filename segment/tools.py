#!/usr/bin/env python
import numpy as np


def make_bbox(feats, rmax, cmax, pad):
    '''Make bounding boxes for labelled regions
       ----------------------------------------
       Make a list of bounding box coordinates from
       region properties list or np.array

       Parameters
       ----------
       feats : np.array or list
           List of RegionProperties or np.array with
           bounding box coordinates
       rmax : int
           Maximum height of an image
       cmax : int
           Maximum width of an image
       pad : int
           Padding value for the bounding boxes

       Returns
       -------
       bbox : list
           List of bounding box coordinates in
           (xmin, xmax, ymin, ymax) tuples

    '''
    bbox = []
    for i in range(len(feats)):
        if type(feats) == np.ndarray:
            ymin, xmin, ymax, xmax = feats[i]
        elif type(feats) == list:
            ymin, xmin, ymax, xmax = feats[i].bbox
        bb = np.array((max(0, xmin - pad),
                       min(xmax + pad, cmax - 1),
                       max(0, ymin - pad),
                       min(ymax + pad, rmax - 1)))
        bbox.append(bb)
    return bbox


def read_bbox(df, rmax, cmax, columns=['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3'],
              pad=0):
    '''Extract bounding boxes from DataFrame
       -------------------------------------
       Make a list of bounding box coordinate tuples
       from the region properties DataFrame

       Parameters
       ----------
       df : DataFrame
           DataFrame of RegionProperties
       rmax : int
           Maximum height of an image
       cmax : int
           Maximum width of an image
       pad : int
           Padding value for the bounding boxes

       Returns
       -------
       bbox : list
           List of bounding box coordinates in
           (xmin, xmax, ymin, ymax) tuples
    '''
    bbox_array = df[columns].values

    bbox = make_bbox(feats=bbox_array,
                     rmax=rmax, cmax=cmax,
                     pad=pad)
    return bbox
