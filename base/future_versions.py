#!/usr/bin/env python3
import numpy as np
from skimage.measure import regionprops

OBJECT_COLUMNS = {
    'image', 'coords', 'convex_image', 'slice',
    'filled_image', 'intensity_image'
}

COL_DTYPES = {
    'area': int,
    'bbox': int,
    'bbox_area': int,
    'moments_central': float,
    'centroid': float,
    'convex_area': int,
    'convex_image': object,
    'coords': object,
    'eccentricity': float,
    'equivalent_diameter': float,
    'euler_number': int,
    'extent': float,
    'filled_area': int,
    'filled_image': object,
    'moments_hu': float,
    'image': object,
    'inertia_tensor': float,
    'inertia_tensor_eigvals': float,
    'intensity_image': object,
    'label': int,
    'local_centroid': float,
    'major_axis_length': float,
    'max_intensity': int,
    'mean_intensity': float,
    'min_intensity': int,
    'minor_axis_length': float,
    'moments': float,
    'moments_normalized': float,
    'orientation': float,
    'perimeter': float,
    'slice': object,
    'solidity': float,
    'weighted_moments_central': float,
    'weighted_centroid': float,
    'weighted_moments_hu': float,
    'weighted_local_centroid': float,
    'weighted_moments': float,
    'weighted_moments_normalized': float
}


def _props_to_dict(regions, properties=('label', 'bbox'), separator='-'):
    out = {}
    n = len(regions)
    for prop in properties:
        dtype = COL_DTYPES[prop]
        column_buffer = np.zeros(n, dtype=dtype)
        r = regions[0][prop]

        # scalars and objects are dedicated one column per prop
        # array properties are raveled into multiple columns
        # for more info, refer to notes 1
        if np.isscalar(r) or prop in OBJECT_COLUMNS:
            for i in range(n):
                column_buffer[i] = regions[i][prop]
            out[prop] = np.copy(column_buffer)
        else:
            if isinstance(r, np.ndarray):
                shape = r.shape
            else:
                shape = (len(r),)

            for ind in np.ndindex(shape):
                for k in range(n):
                    loc = ind if len(ind) > 1 else ind[0]
                    column_buffer[k] = regions[k][prop][loc]
                modified_prop = separator.join(map(str, (prop,) + ind))
                out[modified_prop] = np.copy(column_buffer)
    return out


def regionprops_table(label_image, intensity_image=None,
                      properties=('label', 'bbox'),
                      cache=True, separator='-'):

    regions = regionprops(label_image,
                          intensity_image=intensity_image,
                          cache=cache)

    if len(regions) == 0:
        label_image = np.zeros((3,) * label_image.ndim, dtype=int)
        label_image[(1,) * label_image.ndim] = 1
        if intensity_image is not None:
            intensity_image = np.zeros(label_image.shape,
                                       dtype=intensity_image.dtype)
        regions = regionprops(label_image, intensity_image=intensity_image,
                              cache=cache)

        out_d = _props_to_dict(regions, properties=properties,
                               separator=separator)
        return {k: v[:0] for k, v in out_d.items()}

    return _props_to_dict(regions, properties=properties, separator=separator)
