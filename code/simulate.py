#!/usr/bin/env python

import nibabel as nib
import numpy as np

import metric
import noise


def load_data():
    return {"id1": 'array1',
            "id2": 'array2'}


def main():
    datasets = load_data()

    metrics = {"MSE": metric.mse,
               "Structured Sim.": metric.ssim}
    noises = {"1-Voxel": [noise.oneVoxel, [30, 30, 30], .1]}

    noisy_datasets = []
    for did in datasets.keys():
        for nid in noises.keys():
            noisy_datasets += [{"dset": did,
                                "noise": nid,
                                "image": noise.applyNoise(datasets[did],
                                                          *noises[nid])}]

if __name__ == "__main__":
    main()
