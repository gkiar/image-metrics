#!/usr/bin/env python

import numpy as np
from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as nrmse
from scipy.stats import entropy


def nmi(A, B, bins=1000):
    """Compute the normalized mutual information.

    The normalized mutual information is given by:

                H(A) + H(B)
      Y(A, B) = -----------
                  H(A, B)

    where H(X) is the entropy ``- sum(x log x) for x in X``.

    Parameters
    ----------
    A, B : ndarray
        Images to be registered.

    Returns
    -------
    nmi : float
        The normalized mutual information between the two arrays, computed at a
        granularity of 100 bins per axis (10,000 bins total).
    """
    hist, bins_A, bins_B = np.histogram2d(np.sort(np.ravel(A)),
                                                  np.sort(np.ravel(B)),
                                                  bins=bins)
    hist /= np.sum(hist)

    H_A = entropy(np.sum(hist, axis=0))
    H_B = entropy(np.sum(hist, axis=1))
    H_AB = entropy(np.ravel(hist))

    return (H_A + H_B) / H_AB


def cnrmse(A, B):
    return np.sqrt( nrmse(A, B) * nrmse(B, A) )
