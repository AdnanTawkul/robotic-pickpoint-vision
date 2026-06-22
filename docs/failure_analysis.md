# Failure-Case Analysis

This document will be expanded throughout the project.

## Initial expected failure modes

- Low contrast between object and background
- Strong shadows or highlights
- Motion blur or defocus blur
- Partial occlusion
- Cluttered scenes with touching objects
- Symmetric objects where orientation is visually ambiguous
- Detection failure on object categories not represented in training or examples

## How we will analyze failures

For each failure case, we will document:

- input image
- expected behavior
- actual behavior
- suspected cause
- possible fix
- whether the issue is acceptable for a demo portfolio project

## Step 7 robustness cases

Step 7 introduces controlled visual degradations. These are not yet full quantitative failures; they are visual stress-test inputs that will be used by the next evaluation step.

### Blur

Expected risk: contour boundaries become softer, which can shift segmentation and orientation estimation.

### Noise

Expected risk: thresholding and contour extraction may detect small noisy blobs unless preprocessing is robust.

### Brightness changes

Expected risk: very dark or very bright images may reduce foreground/background separability.

### Contrast changes

Expected risk: low contrast can make segmentation unstable, while very high contrast can amplify artifacts.

### Partial occlusion

Expected risk: the visible contour may no longer represent the full object geometry, causing center and orientation estimates to shift.
