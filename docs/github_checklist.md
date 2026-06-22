# GitHub Publishing Checklist

Before publishing the repository:

## Files to commit

Commit:

- source code under `src/`
- scripts under `scripts/`
- tests under `tests/`
- Streamlit app under `app/`
- README and docs
- selected safe screenshots under `docs/assets/`

## Files not to commit

Do not commit:

- `.venv/`
- generated `outputs/` files
- generated `data/synthetic/` files
- private phone images
- downloaded model weights such as `yolov8n.pt`
- large exported models such as `.onnx` or `.engine`

## Suggested final commits

1. `Add Streamlit pick-point demo app`
2. `Improve Streamlit image display sizing`
3. `Polish README and project documentation`

## Optional screenshots

Run:

```powershell
py scripts\export_demo_assets.py
```

Then review:

```text
docs/assets/
```

Only commit screenshots that are safe to publish.

## GitHub repository settings

Recommended:

- Add a concise repository description:
  `Vision-based pick-point and orientation estimation for robotic manipulation using OpenCV, YOLO, synthetic data, robustness tests, and Streamlit.`
- Add topics:
  `computer-vision`, `opencv`, `yolo`, `robotics`, `image-processing`, `streamlit`, `pytorch`, `synthetic-data`
- Add a project screenshot to the README.
- Pin the repository on your GitHub profile.
