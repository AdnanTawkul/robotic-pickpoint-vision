# Step 13C: Mouse Detection Tips and Streamlit UI Cleanup

## Why the mouse became worse

The mouse is actually a COCO object class, and the earlier baseline detected it at a normal confidence. In the screenshot, the confidence threshold is set to `0.95`, which is too strict for this image. That can suppress the YOLO mouse detection and force the OpenCV fallback or a poor detection path.

## Recommended mouse settings

Use the Streamlit preset:

```text
Mouse / common COCO object
```

Or manually use:

```text
YOLO model: yolov8n.pt
Confidence: 0.20 to 0.30
Image size: 960
Augmentation: off
Class filter: mouse
Maximum detections: 3
Fallback: on
```

## Recommended CLI command

```powershell
py scripts\run_yolo_detection.py --input-dir data\sample_images --model yolov8n.pt --confidence 0.20 --img-size 960 --classes mouse
```

Integrated pipeline:

```powershell
py scripts\run_integrated_pipeline.py --input-dir data\sample_images --model yolov8n.pt --confidence 0.20 --img-size 960 --classes mouse
```

## UI changes

This patch replaces the oversized `Mode` metric with compact readable text and adds detection presets:

- Balanced default
- Sensitive tabletop
- Mouse / common COCO object
- Scissors / thin tools
