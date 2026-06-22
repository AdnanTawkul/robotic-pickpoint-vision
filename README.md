# Vision-Based Pick-Point Estimation for Robotic Manipulation

A practical computer-vision project for estimating 2D object pick points and orientations from RGB images.

Repository name: `robotic-pickpoint-vision`

## Recruiter summary

This project demonstrates a robotics-style perception pipeline without requiring a robot arm, depth camera, or external hardware.

The system can:

- generate synthetic pick-object scenes with ground-truth labels
- estimate object center and orientation using OpenCV contours, PCA, and `minAreaRect`
- evaluate center-point error, orientation error, pass rates, and inference speed
- stress-test the pipeline under blur, noise, brightness, contrast, and occlusion
- run YOLO object detection as an optional front end
- combine YOLO detections with OpenCV pose estimation and fallback segmentation
- provide an interactive Streamlit demo for image upload and visualization

This repository is intended to show practical computer vision, error analysis, robustness testing, and clean Python software engineering.

---

## Demo preview

After generating demo assets, screenshots can be shown here:

```powershell
py scripts\export_demo_assets.py
```

Expected exported assets:

```text
docs/assets/synthetic_demo_grid.png
docs/assets/robustness_evaluation_grid.png
docs/assets/integrated_pickpoint_grid.png
```

Recommended README screenshots:

| Synthetic pose estimation | Robustness evaluation | Integrated real-image pipeline |
|---|---|---|
| `docs/assets/synthetic_demo_grid.png` | `docs/assets/robustness_evaluation_grid.png` | `docs/assets/integrated_pickpoint_grid.png` |

> Before committing real-image screenshots, make sure the images are safe to publish.

---

## Pipeline overview

```text
Input image
   |
   |-- Optional YOLO detection
   |       |
   |       └── object bounding boxes
   |
   |-- OpenCV segmentation
   |       |
   |       └── binary foreground mask
   |
   |-- Contour extraction
   |       |
   |       ├── center estimation
   |       ├── orientation estimation
   |       └── pick-point recommendation
   |
   └── annotated output + CSV/JSON metrics
```

The project supports two complementary perception paths:

1. **Synthetic/evaluation path**  
   Uses generated masks and labels to measure accuracy against known ground truth.

2. **Real-image demo path**  
   Uses YOLO when available and falls back to OpenCV segmentation when YOLO fails or is not useful.

---

## Key features

- Synthetic rotated-object dataset generation
- Ground-truth labels in CSV and JSON
- OpenCV contour-based center estimation
- PCA-based orientation estimation
- `minAreaRect` orientation baseline
- Pick point visualization
- Annotated output image saving
- CSV, JSON, and Markdown evaluation reports
- Robustness tests for:
  - blur
  - noise
  - brightness
  - contrast
  - partial occlusion
- YOLO detection path using Ultralytics
- Integrated YOLO + OpenCV pipeline
- Streamlit GUI for upload-based demos
- Unit tests for core modules

---

## Setup

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
```

Install the project:

```powershell
py -m pip install --upgrade pip
py -m pip install -e .
```

For YOLO support:

```powershell
py -m pip install ultralytics
```

---

## Quick start

Run the full synthetic demo:

```powershell
py scripts\run_demo.py --regenerate
```

Open:

```text
outputs/annotated/demo/demo_grid.png
```

Run the Streamlit app:

```powershell
streamlit run app\streamlit_app.py
```

Run all tests:

```powershell
py -m pytest
```

---

## Main commands

### Generate synthetic dataset

```powershell
py scripts\create_synthetic_dataset.py --num-images 20 --clear
```

Outputs:

```text
data/synthetic/images/
data/synthetic/masks/
data/synthetic/labels.csv
data/synthetic/labels.json
data/synthetic/preview_grid.png
```

### Run synthetic pose-estimation demo

```powershell
py scripts\run_demo.py --regenerate
```

Outputs:

```text
outputs/annotated/demo/demo_grid.png
outputs/metrics/demo_summary.json
```

### Run evaluation

```powershell
py scripts\evaluate.py
```

Outputs:

```text
outputs/metrics/evaluation/per_sample_metrics.csv
outputs/metrics/evaluation/evaluation_summary.json
outputs/metrics/evaluation/evaluation_report.md
```

### Run robustness evaluation

```powershell
py scripts\create_robustness_variants.py --clear
py scripts\evaluate_robustness.py
```

Outputs:

```text
outputs/robustness/step7/
outputs/metrics/robustness/per_variant_metrics.csv
outputs/metrics/robustness/robustness_summary.json
outputs/metrics/robustness/robustness_report.md
outputs/metrics/robustness/robustness_evaluation_grid.png
```

### Run YOLO detection

```powershell
py scripts\run_yolo_detection.py --input-dir data\sample_images
```

Outputs:

```text
outputs/annotated/yolo/yolo_detection_grid.png
outputs/metrics/yolo_detections.csv
```

### Run integrated pick-point pipeline

For real images:

```powershell
py scripts\run_integrated_pipeline.py --input-dir data\sample_images
```

For deterministic OpenCV-only testing:

```powershell
py scripts\run_integrated_pipeline.py --input-dir data\synthetic\images --no-yolo --max-images 5
```

Outputs:

```text
outputs/annotated/integrated/integrated_pickpoint_grid.png
outputs/metrics/integrated_pickpoints.csv
outputs/metrics/integrated_pickpoints.json
```

### Run Streamlit app

```powershell
streamlit run app\streamlit_app.py
```

The app supports:

- image upload
- YOLO + OpenCV fallback mode
- OpenCV-only mode
- YOLO confidence threshold
- optional YOLO class filter
- image display-width control
- annotated image download
- result CSV download

---

## Results

### Synthetic clean-mask baseline

On the default synthetic dataset, the mask-based baseline typically achieves:

| Metric | Result |
|---|---:|
| Mean center error | 0.334 px |
| Max center error | 0.805 px |
| Mean PCA orientation error | 0.832 deg |
| P90 PCA orientation error | 2.166 deg |
| Mean `minAreaRect` orientation error | 2.776 deg |
| Center pass rate at 2 px | 100% |
| Orientation pass rate at 5 deg | 100% |

### Robustness evaluation

The image-based OpenCV path is robust to blur, brightness, contrast, and noise on the synthetic scenes. Partial occlusion is the dominant failure case.

Example grouped results:

| Transform | Observation |
|---|---|
| Blur | Stable center and orientation estimation |
| Noise | Stable after morphology cleanup |
| Brightness | Stable on synthetic contrast |
| Contrast | Stable unless foreground/background separation becomes weak |
| Occlusion | Large center and orientation errors |

### Real-image integrated pipeline

The real-image integrated pipeline works as a demonstration of YOLO + geometric fallback. It can estimate pick points on user-provided images, but quality depends on segmentation quality and object/background contrast.

Observed limitations:

- YOLO may miss unusual objects or objects outside COCO-like categories.
- OpenCV fallback can include background when object and background are visually similar.
- Thin or reflective objects may produce unstable contours.
- Symmetric objects can have ambiguous orientation.

These limitations are documented intentionally because practical perception projects should show both successful cases and failure cases.

---

## Testing

Run:

```powershell
py -m pytest
```

Expected after Step 11:

```text
25 passed
```

---

## Project structure

```text
robotic-pickpoint-vision/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   └── pickpoint_vision/
│       ├── app_utils.py
│       ├── detection.py
│       ├── evaluation.py
│       ├── integrated_pipeline.py
│       ├── pipeline.py
│       ├── pose_estimation.py
│       ├── robustness.py
│       ├── robustness_evaluation.py
│       ├── segmentation.py
│       ├── synthetic_data.py
│       ├── utils.py
│       └── visualization.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── create_synthetic_dataset.py
│   ├── create_robustness_variants.py
│   ├── evaluate.py
│   ├── evaluate_robustness.py
│   ├── export_demo_assets.py
│   ├── run_demo.py
│   ├── run_integrated_pipeline.py
│   ├── run_pose_estimation.py
│   ├── run_yolo_detection.py
│   └── visualize_synthetic_results.py
├── tests/
├── data/
├── outputs/
└── docs/
```

---

## What this project demonstrates

- Practical image-processing pipeline design
- OpenCV contour and PCA pose estimation
- Synthetic data generation with known labels
- Quantitative evaluation rather than only visual demos
- Robustness and failure-case analysis
- YOLO integration
- Streamlit app development
- Clean Python packaging and testing
- GitHub-ready documentation

---

## Future improvements

- Train a custom detector on industrial pick objects
- Add segmentation models such as YOLO segmentation or SAM-style masks
- Improve background subtraction for dark objects on dark surfaces
- Add multiple-object synthetic scenes
- Add automatic failure-case ranking
- Export ONNX models for deployment-style demos
- Add optional depth simulation from monocular cues or synthetic data

---

## License

Add a license before publishing if you want others to reuse this code.
