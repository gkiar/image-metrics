#!/usr/bin/env python

import numpy as np


def applyNoise(data, fn=None, *args):
    if fn is not None:
        dataset = fn(dataset, *args)
    return dataset


def oneVoxel(image, loc=[0,0,0], scale=0.1):
    image[loc[0], loc[1], loc[2]] = image[loc[0], loc[1], loc[2]] * (1 + scale)
    return image
