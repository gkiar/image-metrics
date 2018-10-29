#!/usr/bin/env python

from argparse import ArgumentParser
from copy import deepcopy
import pandas as pd
import numpy as np
import os.path as op
import os

import boutiques.creator as bc
from bids.layout import BIDSLayout
import nibabel as nib

import metrics
import noise


def loadImage(path):
    im = nib.load(path)
    dat = im.get_data()
    return dat


def makeParser():
    parser = ArgumentParser("noise_simulations.py", description="A script which"
                            " accepts a BIDS dataset, modality, and optionally "
                            "skull stripped images, then adds noise and "
                            "compares various metrics for characterizing the "
                            "noise.")
    parser.add_argument("bids_dataset", action="store",
                        help="")
    parser.add_argument("modality", action="store",
                        choices=["anat", "func", "dwi"],
                        help="")
    parser.add_argument("--skull_stripped", "-s", action="store",
                        help="")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="")
    return parser

def main():
    parser = makeParser()
    descriptor = bc.CreateDescriptor(parser, execname="noise_simulations.py")
    descriptor.save("noise_simulations.json")

    results = parser.parse_args()
    verb = results.verbose
    cwd = os.getcwd()

    # Get files of the specified modality
    bids = BIDSLayout(results.bids_dataset)
    dataset_name = bids.description['Name']
    mod = results.modality
    modalities = bids.get_modalities()
    if mod not in modalities:
        print("Modality not found. Choose one of {0}".format(modalities))
        return -1
    else:
        files = bids.get(modality=mod, return_type='file')
        if verb:
            print("Files found: {0}".format(", ".join(f for f in files)))
        # In pybids 0.7 this will be datatype, not modality

    # Setup pandas tables for images and distances
    distance_df = pd.DataFrame(columns = ["Filename", "Noise",
                                          "Metric", "Value"])
                                      #  "CC", "MSE", "CNRMSE", "SSIM", "NMI"
    image_df = pd.DataFrame(columns = ["Filename", "Noise", "NoiseParams"])

    images_f = op.join(cwd, "{0}_{1}_images.tsv".format(dataset_name, mod))
    distances_f = op.join(cwd,
                          "{0}_{1}_distances.tsv".format(dataset_name, mod))

    # Setup noise parameters for 1-Voxel
    loc = "middle"
    scale_1vox = 1.5

    # Setup noise parameters for Rician
    b = 1.1
    scale_ric = 1.5

    # Setup noise parameters for Gaussian
    scale_gau = 1.2

    # Process data....
    for fil in files:
        if verb:
            print("Processing file: {0} ...".format(fil))
        # Load the original image
        image_raw = loadImage(fil)
        image_df.loc[len(image_df)] = [fil, None, None]

        # Apply 1-voxel noise
        if loc == "middle":
            loc_1vox = tuple(int(v/2 - 3 + 6*np.random.random(1))
                             for v in np.asarray(image_raw.shape))
        if verb:
            print("... Applying 1-voxel noise at {0}".format(loc_1vox))
        image_1vox = deepcopy(image_raw)
        image_1vox = noise.oneVoxelNoise(image_1vox, loc_1vox, scale_1vox)
        image_df.loc[len(image_df)] = [fil, "1-Voxel", [loc_1vox, scale_1vox]]

        # Apply Rician noise
        if verb:
            print("... Applying Rician noise with b = {0}".format(b))
        image_ric = deepcopy(image_raw)
        image_ric = noise.ricianNoise(image_ric, b, scale_ric)
        image_df.loc[len(image_df)] = [fil, "Rician", [b, scale_ric]]

        # Compute distances between images
        if verb:
            print("... Computing distances between images")

        # Computing 1-voxel distances
        distance_df.loc[len(distance_df)] = [fil, "1-voxel", "CC",
                                             1 - metrics.cc(image_raw,
                                                            image_1vox)]
        distance_df.loc[len(distance_df)] = [fil, "1-voxel", "MSE",
                                             metrics.mse(image_raw, image_1vox)]
        distance_df.loc[len(distance_df)] = [fil, "1-voxel", "CNRMSE",
                                             metrics.cnrmse(image_raw,
                                                            image_1vox)]
        distance_df.loc[len(distance_df)] = [fil, "1-voxel", "SSIM",
                                             1 - metrics.ssim(image_raw,
                                                              image_1vox)]
        distance_df.loc[len(distance_df)] = [fil, "1-voxel", "NMI",
                                             1 - metrics.nmi(image_raw,
                                                             image_1vox)]

        #Computing Rician distances
        distance_df.loc[len(distance_df)] = [fil, "Rician", "CC",
                                             1 - metrics.cc(image_raw,
                                                            image_ric)]
        distance_df.loc[len(distance_df)] = [fil, "Rician", "MSE",
                                             metrics.mse(image_raw, image_ric)]
        distance_df.loc[len(distance_df)] = [fil, "Rician", "CNRMSE",
                                             metrics.cnrmse(image_raw,
                                                            image_ric)]
        distance_df.loc[len(distance_df)] = [fil, "Rician", "SSIM",
                                             1 - metrics.ssim(image_raw,
                                                              image_ric)]
        distance_df.loc[len(distance_df)] = [fil, "Rician", "NMI",
                                             1 - metrics.nmi(image_raw,
                                                             image_ric)]

    if verb:
        print("Saving image data to {0} ...".format(images_f))
        print(image_df)

    with open(images_f, "w") as fhandle:
        image_df.to_csv(fhandle, sep="\t", index=False)

    if verb:
        print("Saving distance data to {0} ...".format(distances_f))
        print(distance_df)
    with open(distances_f, "w") as fhandle:
        distance_df.to_csv(fhandle, sep="\t", index=False)


if __name__ == "__main__":
    main()

"""
data = []

samesubj_df = diff_df.loc[diff_df.SameSubject.values == True]
samescan_df = samesubj_df.loc[samesubj_df.SameSession.values == True]

df = samescan_df
comp = ['1-voxel', 'Rician', 'Gaussian']
for c in comp:
    notc = deepcopy(comp)
    notc.remove(c)

    series = df[df[c] == True][df[notc[0]] == False][df[notc[1]] == False].SSIM

    trace = {
        "type": "violin",
        "x": [c] * len(series),
        "y": series,
        "name": c,
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }

    data.append(trace)

fig = {
    "data": data,
    "layout": {
        "title": "",
        "yaxis": {
            "zeroline": False,
            "title": "SSIM"
        }
    }
}

iplot(fig, validate=False)
"""
