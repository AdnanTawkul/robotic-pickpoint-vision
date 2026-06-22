# Detection Improvement Plan

## Step 13A: YOLO inference upgrade

This step improves off-the-shelf YOLO usage without training a custom model.

New controls:

- model size, for example `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`
- confidence threshold
- inference image size
- IoU threshold
- test-time augmentation
- max detections

Recommended difficult-image command:

```powershell
py scripts\run_yolo_detection.py --input-dir data\sample_images --model yolov8s.pt --confidence 0.10 --img-size 1280 --augment
```

Recommended integrated command:

```powershell
py scripts\run_integrated_pipeline.py --input-dir data\sample_images --model yolov8s.pt --confidence 0.10 --img-size 1280 --augment
```

## Why this helps

- Larger YOLO models improve recognition quality.
- Larger image size helps small objects.
- Lower confidence can expose weak detections.
- Test-time augmentation can improve difficult angles but is slower.

## Tradeoff

This can improve detection, but it still cannot guarantee detection of objects outside the model's training distribution.

## Next step

Step 13B will add OpenCV object proposals so the pipeline can find object-like regions even when YOLO misses them.
