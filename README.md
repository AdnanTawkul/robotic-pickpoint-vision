# Vision-Based Pick-Point Estimation for Robotic Manipulation

A practical computer-vision portfolio project for estimating object pick points and orientations from 2D images.

Repository name: `robotic-pickpoint-vision`

## Project goal

Build a no-hardware perception pipeline that detects objects, estimates a recommended pick point, estimates object orientation, visualizes the result, and evaluates accuracy and robustness.

## Current status

Completed:

- repository foundation
- synthetic dataset generation
- OpenCV pose estimation from masks
- visualization utilities
- first end-to-end command-line demo

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

## Run the current demo

```powershell
py scripts\run_demo.py --regenerate
```

The demo creates annotated outputs here:

```text
outputs/annotated/demo/
```

Open this file to inspect the visual result:

```text
outputs/annotated/demo/demo_grid.png
```

The demo summary is saved here:

```text
outputs/metrics/demo_summary.json
```

## Run tests

```powershell
py -m pytest
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
│       ├── pipeline.py
│       ├── pose_estimation.py
│       ├── synthetic_data.py
│       ├── utils.py
│       └── visualization.py
├── scripts/
│   ├── create_synthetic_dataset.py
│   ├── run_demo.py
│   ├── run_pose_estimation.py
│   ├── verify_setup.py
│   └── visualize_synthetic_results.py
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
