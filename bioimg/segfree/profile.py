#!/usr/bin/env python3
import numpy as np
import pandas as pd


def tile_image(arr, height, width):
    """Tile image into blocks of specified width / height
    """
    h, w = arr.shape
    assert h % height == 0, "Image of height {} is not evenly divisble by {}".format(
        h, height)
    assert w % width == 0, "Image of width {} is not evenly divisble by {}".format(
        w, width)
    return (arr.reshape(h//height, height, -1, width)
               .swapaxes(1, 2)
               .reshape(-1, height, width))


def get_blocktype(a, nbits=8):
    block_type = {i : np.sum(a == i) / a.size for i in range(nbits)}
    return pd.DataFrame([block_type])
