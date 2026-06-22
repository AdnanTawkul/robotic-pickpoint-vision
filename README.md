# Vision-Based Pick-Point Estimation for Robotic Manipulation

A practical computer-vision portfolio project for estimating object pick points and orientations from 2D images.

Repository name: `robotic-pickpoint-vision`

## Project goal

Build a no-hardware perception pipeline that detects objects, estimates a recommended robotic pick point, estimates object orientation, visualizes the result, and evaluates accuracy and robustness.

## Step 1 status

This repository currently contains the initial project structure, setup files, documentation placeholders, and a setup verification script.

## Planned final features

- Input image or folder of images
- Object detection using YOLO or classical/synthetic-object detection where appropriate
- Center-point estimation
- 2D orientation estimation using OpenCV contour analysis, PCA, and/or `minAreaRect`
- Annotated visual output with bounding box, contour, center point, orientation axis, and pick point
- Saved result images
- Evaluation metrics:
  - center-point error
  - orientation error
  - detection confidence summary
  - inference speed
- Robustness tests:
  - blur
  - noise
  - brightness changes
  - contrast changes
  - partial occlusion
- Streamlit demo app
- Failure-case analysis

## Setup

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
```

Install the project in editable mode:

```powershell
py -m pip install --upgrade pip
py -m pip install -e .
```

Verify the Step 1 setup:

```powershell
py scripts\verify_setup.py
```

Expected result:

```text
Step 1 setup verification passed.
```

## Current folder structure

```text
robotic-pickpoint-vision/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   └── pickpoint_vision/
│       ├── __init__.py
│       └── utils.py
├── scripts/
│   └── verify_setup.py
├── app/
├── tests/
├── data/
│   ├── sample_images/
│   └── synthetic/
├── outputs/
│   ├── annotated/
│   └── metrics/
└── docs/
    ├── project_plan.md
    └── failure_analysis.md
```

## License

Add a license later if you want this repository to be fully reusable by others.
