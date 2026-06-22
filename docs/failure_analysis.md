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

## Step 8 quantitative robustness evaluation

Step 8 evaluates degraded RGB images using classical image-based segmentation. This is intentionally harder than the clean-mask baseline.

Expected observations:

- blur should usually remain stable for large objects
- mild noise should usually remain stable after morphology cleanup
- strong noise may create extra foreground regions
- low contrast can reduce segmentation reliability
- partial occlusion may shift the estimated contour, center, and orientation

These are useful portfolio discussion points because they show practical error analysis instead of only ideal-case demos.

## Step 9 YOLO observations

YOLO detects common COCO-like objects well, but it can fail on domain-specific or unusual objects such as a stylized bottle, a toy car, or a computer mouse viewed from unusual angles. It can also assign incorrect COCO classes to synthetic geometric objects.

This is acceptable for the project because Step 9 demonstrates the detection interface, while Step 10 integrates detection with classical geometric pose estimation and fallback segmentation.

## Step 10 integrated-pipeline limitations

The integrated pipeline uses classical segmentation inside YOLO boxes or across the full image. It can struggle when:

- object and background have similar color or brightness
- the object is dark on a dark background
- the YOLO box includes too much background
- the object has holes, thin structures, or multiple disconnected components
- the YOLO class is wrong but the box still overlaps the object

These are realistic limitations and motivate future improvements such as trained segmentation masks, SAM-style segmentation, or custom YOLO training on industrial pick objects.
