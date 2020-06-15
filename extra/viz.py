#!/usr/bin/env python3
"""
Plotly visualization of annotated images
"""

import plotly.graph_objs as go
import numpy as np
from PIL import Image
from skimage.util import img_as_ubyte


def make_layout(img, scale_factor):
    '''Make plotly layout
       ------------------
       Generates a plotly Layout object from
       the RGB or grayscale image

       Parameters
       ----------
       img : array
           Intensity image or RGB overlay
       scale_factor : float
           The scale factor controls the default anchor view

       Returns
       -------
       layout : Layout object
           Plotly Layout object can be used to
           generate an interactive visualization
    '''
    im = Image.fromarray(img_as_ubyte(img))
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, img.shape[1]*scale_factor]
        ),
        yaxis=go.layout.YAxis(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, img.shape[0]*scale_factor],
            scaleanchor='x'
        ),
        autosize=False,
        height=img.shape[0]*scale_factor,
        width=img.shape[1]*scale_factor,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        images=[go.layout.Image(
            x=0,
            sizex=img.shape[1]*scale_factor,
            y=img.shape[0]*scale_factor,
            sizey=img.shape[0]*scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=im)]
    )
    return layout


def plotly_viz(img, bb, scale_factor=0.5):
    '''Get Layout and Scatter objects
       ------------------------------
       Generate Layout using the provided image and
       interactive bounding boxes (Scatter object)

       Parameters
       ----------
       img : array
           Intensity image or RGB overlay
       bb : list
           List of bounding box coordinate tuples
       scale_factor : float
           The scale factor controls the default anchor view

       Returns
       -------
       layout : Layout object
           Plotly Layout object can be used to
           generate an interactive visualization
       boxviz : Scatter object
           Interactive visualization of bounding boxes
    '''
    layout = make_layout(img=img, scale_factor=scale_factor)
    # scale the bounding boxes
    bb_scaled = [np.array(scale_factor*b, dtype=np.int) for b in bb]

    boxviz = [
        go.Scatter(
            x=[b[0], b[1], b[1], b[0], b[0]],
            y=[int(img.shape[0]*scale_factor) - b[3],
               int(img.shape[0]*scale_factor) - b[3],
               int(img.shape[0]*scale_factor) - b[2],
               int(img.shape[0]*scale_factor) - b[2],
               int(img.shape[0]*scale_factor) - b[3]],
            hoveron='fills',
            name='Box',
            text=str(i),
            mode='lines',
            line=dict(width=3, color='white'),
            showlegend=False
        )
        for b, (i, _) in zip(bb_scaled, enumerate(bb_scaled))]

    return layout, boxviz


def plotly_predictions(img, bb, ypred, labels, scale_factor=0.5):
    '''Get Layout and Scatter objects for predictions
       ----------------------------------------------
       Generate Layout using the provided image and
       interactive bounding boxes (Scatter object) and
       show predicted labels

       Parameters
       ----------
       img : array
           Intensity image or RGB overlay
       bb : list
           List of bounding box coordinate tuples
       ypred : array
           Integer-valued array with predicted class
           for each instance
       labels : list or array
           Class names / labels
       scale_factor : float
           The scale factor controls the default anchor view

       Returns
       -------
       layout : Layout object
           Plotly Layout object can be used to
           generate an interactive visualization
       boxviz : Scatter object
           Interactive visualization of bounding boxes
    '''
    layout = make_layout(img=img, scale_factor=scale_factor)

    # scale the bounding boxes
    bb_scaled = [np.array(scale_factor*b, dtype=np.int) for b in bb]

    boxviz = [
        go.Scatter(
            x=[b[0], b[1], b[1], b[0], b[0]],
            y=[int(img.shape[0]*scale_factor) - b[3],
               int(img.shape[0]*scale_factor) - b[3],
               int(img.shape[0]*scale_factor) - b[2],
               int(img.shape[0]*scale_factor) - b[2],
               int(img.shape[0]*scale_factor) - b[3]],
            hoveron='fills',
            name='Box',
            text=labels[y] + " | " + str(i),
            mode='lines',
            line=dict(width=3, color='white'),
            showlegend=False
        )
        for b, y, (i, _) in zip(bb_scaled, ypred, enumerate(bb_scaled))]

    return layout, boxviz


def update_feats(img, bb, y_pred, target_names, scale_factor=0.5):
    bb_scaled = [np.array(scale_factor*b, dtype=np.int) for b in bb]

    feats = [
        go.Scatter(
            x=[b[0], b[1], b[1], b[0], b[0]],
            y=[int(img.shape[0]*scale_factor) - b[3],
               int(img.shape[0]*scale_factor) - b[3],
               int(img.shape[0]*scale_factor) - b[2],
               int(img.shape[0]*scale_factor) - b[2],
               int(img.shape[0]*scale_factor) - b[3]],
            hoveron='fills',
            name='Sampled features',
            text=target_names[y] + " | " + str(i),
            mode='lines',
            line=dict(width=3, color='white'),
            showlegend=False
        )
        for b, y, (i, _) in zip(bb_scaled, y_pred, enumerate(bb_scaled))]

    return feats
