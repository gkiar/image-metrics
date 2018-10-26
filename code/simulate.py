#!/usr/bin/env python

from copy import deepcopy
import pandas as pd
import numpy as np

from pybids.layout import BIDSLayout
import nibabel as nib

import metrics
import noise


def loadImage(path):
    im = nib.load(path)
    dat = im.get_data()
    return im


def main():
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
    results = parser.parse_args()

    # Get files of the specified modality
    bids = BIDSLayout(results.bids_dataset)
    modalities = bids.get_modalities()
    if results.modality not in modalities:
        print("Modality not found. Choose one of {0}".format(modalities))
        return -1
    else:
        files = bids.get(modality='anat', return_type='file')
        # In pybids 0.7 this will be datatype, not modality

    # Setup pandas tables for images and distances
    distance_df = pd.DataFrame(columns = ["Filename", "Noise", "CC", "MSE",
                                          "CNRMSE", "SSIM", "NMI"])
    image_df = pd.DataFrame(columns = ["Filename", "Image",
                                       "Noise", "NoiseParams"])

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
        # Load the original image
        image_raw = loadImage(fil)
        image_df.loc[len(image_df)] = [fil, image_raw, None, None]

        # Apply 1-voxel noise
        if loc == "middle":
            loc_1vox = tuple(int(v/2 - 3 + 6*np.random.random(1))
                             for v in np.asarray(d.shape))
        image_1vox = deepcopy(image_raw)
        image_1vox = noise.oneVoxelNoise(image_1vox, loc_1vox, scale_1vox)
        image_df.loc[len(image_df)] = [fil, image_1vox,
                                       "1-Voxel", [loc_1vox, scale_1vox]]

        # Apply Rician noise
        image_ric = deepcopy(image_raw)
        image_ric = noise.ricianNoise(image_ric, b, scale_ric)
        image_df.loc[len(image_df)] = [fil, image_ric,
                                       "Rician", [b, scale_ric]]

        # Compute distances between images
        distance_df.loc[len(distance_df)] = [fil, "1-Voxel",
                                             metrics.cc(image_raw,
                                                        image_1vox),
                                             metrics.mse(image_raw,
                                                         image_1vox),
                                             metrics.cnrmse(image_raw,
                                                            image_1vox),
                                             metrics.ssim(image_raw,
                                                          image_1vox),
                                             metrics.nmi(image_raw,
                                                         image_1vox)]
        distance_df.loc[len(distance_df)] = [fil, "Rician",
                                             metrics.cc(image_raw,
                                                        image_ric),
                                             metrics.mse(image_raw,
                                                         image_ric),
                                             metrics.cnrmse(image_raw,
                                                            image_ric),
                                             metrics.ssim(image_raw,
                                                          image_ric),
                                             metrics.nmi(image_raw,
                                                         image_ric)]

    print(distance_df)


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
