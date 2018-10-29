#!/usr/bin/env python

from scipy.stats import rice, norm
import numpy as np


def oneVoxelNoise(data, loc, scale=1.1):
    """applies 1-voxel scaling to the provided image matrix

    The noise introduced is equivalent to a point magnification at the location
    provided with scaled intensity equal to the value provided (default value is
    an increase in intensity by 10%).

    The location provided must be the same length or 1 fewer than the dimensions
    of the data. When the location provided contains fewer dimensions than the
    data matrix, the location is used to index the first dimensions, and noise
    is applied across the entire last dimension.

    Parameters
    ----------
    data : ndarray
        Image to be perturbed with D dimensions.
    loc : list, tuple
        List of coordinate locations for applying noise. Must be length D or D-1
    scale : float
        Multiplier for the signal at the target location. Default is 1.1

    Returns
    -------
    data : ndarray
        The perturbed data matrix with signal amplification by scale at location
        loc.
    """
    loc = tuple(loc)  # Lists cannot be used to index arrays, but tuples can
    if len(loc) < len(data.shape):  # If fewer dimensions for location than data
        data[loc, :] = data[loc, :] * (scale)  # Apply noise
    else:
        data[loc] = data[loc] * (scale)  # Apply noise
    return data


def ricianNoise(data, b, scale):
    data += scale * rice.rvs(b, size=data.shape)
    return data
