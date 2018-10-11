#!/usr/bin/env python

import nibabel as nib
import numpy as np

import metrics
import noises


def load_data():
    return {"id1": 'array1',
            "id2": 'array2'}


def add_noise(dataset, fn=None):
    if fn is not None:
        dataset = fn(dataset)
    return dataset


def main():
    datasets = load_data()

    metrics = {"MSE": metrics.mse,
               "Structured Sim.": metrics.ssim}
    noises = {"S&P": noises.saltandpepper,
              "Gaussian": noises.gaussian}

    noisy_datasets = []
    for did in datasets.keys():
        for nid in noises.keys():
            noisy_datasets += [{"dset": did,
                                "noise": nid,
                                "image": add_noise(datasets[did], noises[nid])}]


if __name__ == "__main__":
    main()
