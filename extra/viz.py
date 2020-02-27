#!/usr/bin/env python
"""
Plotly visualization of annotated images
"""

import plotly.graph_objs as go
import numpy as np
from PIL import Image
from skimage.util import img_as_ubyte


def plotly_predictions(img, bb, y_pred, target_names, scale_factor=0.5):
    # RGB overlay microscopic image
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

    # scale the bounding boxes
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

    return layout, feats


def plotly_viz(img, bb, scale_factor=0.5):
    # RGB overlay microscopic image
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

    # scale the bounding boxes
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
            text=str(i),
            mode='lines',
            line=dict(width=3, color='white'),
            showlegend=False
        )
        for b, (i, _) in zip(bb_scaled, enumerate(bb_scaled))]

    return layout, feats


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
