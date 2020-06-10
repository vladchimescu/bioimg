#!/usr/bin/env python
import numpy as np
from collections import namedtuple

Box = namedtuple('Box', 'xmin xmax ymin ymax')

def make_bbox(feats, columns, rmax, cmax, pad):
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
            #ymin, xmin, ymax, xmax = feats[i]
            bx = Box(**{k:v for k,v in zip(columns, feats[i])})
        elif type(feats) == list:
            #ymin, xmin, ymax, xmax = feats[i].bbox
            bx = Box(**{k:v for k,v in zip(columns, feats[i].bbox)})
        bb = np.array((max(0, bx.xmin - pad),
                       min(bx.xmax + pad, cmax - 1),
                       max(0, bx.ymin - pad),
                       min(bx.ymax + pad, rmax - 1)))
        bbox.append(bb)
    return bbox


def read_bbox(df, rmax, cmax, columns=['ymin', 'xmin', 'ymax', 'xmax'],
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
       columns : list
           List of column names. Should specify
           the order of xmin, xmax, ymin, ymax
       pad : int
           Padding value for the bounding boxes

       Returns
       -------
       bbox : list
           List of bounding box coordinates in
           (xmin, xmax, ymin, ymax) tuples
    '''
    bbox_array = df[columns].values
    bbox_array = bbox_array.astype(np.int)
    bbox = make_bbox(feats=bbox_array,
                     columns=columns,
                     rmax=rmax, cmax=cmax,
                     pad=pad)
    return bbox
