# Image Quality Metrics for Ultrasound Research (Synthetic Methods Note)

## Contrast-to-noise ratio
CNR = |mean_inside - mean_background| / sqrt(var_inside + var_background). Compute on log-compressed envelope data within matched ROIs placed at the same depth.

## Generalized CNR
gCNR = 1 - overlap of the inside and background intensity histograms; it ranges from 0 to 1 and is robust to dynamic range transformations, which makes it preferred when comparing beamformers or post-processing.

## Speckle SNR
For fully developed speckle, envelope SNR (mean/std) is approximately 1.91. Deviation indicates unresolved scatterers or post-processing smoothing.

## Segmentation metrics
For AI segmentation of structures in ultrasound, report Dice coefficient and 95th-percentile Hausdorff distance (HD95) in millimeters, per-patient, not pooled per-frame, to avoid leakage between train and test frames from the same cine loop.
