# Step 13D: CUDA and General UI Hints

## What changed

This patch removes object-specific tips and makes the Streamlit app more general.

It adds:

- YOLO device selector: `auto`, `cuda:0`, `cpu`
- CUDA-aware YOLO prediction calls
- resolved device display
- brief help text for confidence, image size, IoU, augmentation, class filtering, max detections, and fallback
- general detection presets: Balanced default, Sensitive tabletop, High precision, Small objects

## CUDA behavior

When device is set to `auto`, the code checks PyTorch CUDA availability.

- CUDA available: YOLO receives `device="0"`
- CUDA unavailable: YOLO receives `device="cpu"`

## CLI examples

```powershell
py scripts\run_yolo_detection.py --input-dir data\sample_images --device auto
py scripts\run_integrated_pipeline.py --input-dir data\sample_images --device auto
```

Force CUDA:

```powershell
py scripts\run_yolo_detection.py --input-dir data\sample_images --device cuda:0
```
