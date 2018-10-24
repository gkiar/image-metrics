## Image Metrics

Exploring different distance metrics for comparing images that may or may not be in the same space.


## Ideal Outcome

In order for a metric to be useful, it should properly order images based on their differences. This seems obvious,
but the difficulty arises in that images may not be in the same space/may not overlap, may have different datatypes,
contrasts, or signal to noise ratios... 

This repository will explore several situations for comparing images across these conditions, and will compare various
metrics, and ultimately (hopefully) arrive at a suitable metric to be used in practical solutions where such
comparisons are desired.


## Metric Properties

| Case ID | Desired Property | Example Use Cases | Potential Metrics |
|:--------|:-----------------|:------------------|:------------------|
| 1       | space invariant (translation, rotation, scale, shear, etc.) | comparing unaligned images in different spaces | mutual information |
| 2       | intensity invariant (linear, datatype, other?) | comparing images with different brightnesses or dynamic ranges | structural similarity |
| 3       | space variant    | comparing images aligned to the same space |  structural similarity, correlation coefficient, MSE, NRMSE |
| 4       | intensity variant | comparing | correlation coefficient, MSE, NRMSE |

## Case Studies

#### Case 1:

