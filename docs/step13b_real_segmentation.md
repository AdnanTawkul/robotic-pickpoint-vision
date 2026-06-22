# Step 13B: Real-Object Segmentation and Orientation Fix

## Problem

After Step 13A, YOLO can detect more objects with stronger settings, but the orientation can still be wrong.

Example: the bottle can be detected, but the segmentation inside the YOLO box may become a large rectangular mask. PCA then estimates the orientation of that rectangle or crop instead of the real bottle silhouette.

## Fix

Step 13B adds local-background segmentation:

1. estimate background color from the border of the YOLO region
2. segment pixels that differ from that local background
3. clean the mask with morphology
4. keep the best object-like connected component
5. estimate center and orientation from that component

## Expected improvement

For light objects on dark tabletops, such as the bottle, the orientation line should follow the object body more closely.

## Remaining limitations

- dark objects on dark backgrounds can still be difficult
- transparent or reflective parts may be incomplete
- symmetric objects may not have a meaningful orientation
- wrong YOLO class labels can still happen with off-the-shelf YOLO

## Commands

```powershell
py scripts\apply_step13b_patch.py
py scripts\run_integrated_pipeline.py --input-dir data\sample_images --model yolov8s.pt --confidence 0.10 --img-size 1280 --augment
py -m pytest tests\test_real_segmentation.py
py -m pytest
```
