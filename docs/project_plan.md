# Project Plan

## Project title

Vision-Based Pick-Point Estimation for Robotic Manipulation

## Scope

This project estimates 2D pick points and object orientations from RGB images without requiring a robot arm, depth camera, or external equipment.

## Final demo

The final demo will allow a user to upload an image, run the perception pipeline, and view:

- detected object bounding boxes
- object contours or masks
- center point
- estimated object orientation
- recommended pick point
- inference summary
- saved annotated output image

## Roadmap

1. Define scope and create repository foundation. **Done**
2. Build synthetic image generator with ground-truth labels. **Done**
3. Implement classical OpenCV pose estimation on synthetic objects. **Done**
4. Add visualization utilities. **Done**
5. Add first command-line demo. **Done**
6. Add evaluation metrics. **Current**
7. Add robustness transformations and robustness evaluation.
8. Add YOLO-based object detection path.
9. Integrate detection with pose estimation.
10. Build Streamlit GUI.
11. Add tests, examples, and failure-case analysis.
12. Polish GitHub repository and recruiter-facing README.

## Synthetic dataset design

The synthetic dataset contains one object per image. Each generated image has a matching binary mask and ground-truth label.

Generated files:

```text
data/synthetic/
├── images/
├── masks/
├── labels.csv
├── labels.json
└── preview_grid.png
```

Each label includes:

- image name
- object shape
- center point
- orientation angle in degrees
- pick point
- bounding box
- image dimensions

The first version uses rotated rectangles and ellipses. This is enough to test 2D center and orientation estimation before moving to real images and YOLO.

## Step 3 pose-estimation baseline

Step 3 estimates 2D pose from the synthetic binary masks.

Methods used:

- largest contour extraction with OpenCV
- center estimation using image moments
- orientation estimation using PCA over contour points
- orientation estimation using `cv2.minAreaRect`
- pick point defined as the estimated object center

The output CSV is saved to:

```text
outputs/metrics/pose_estimation_step3.csv
```

This gives us a clean baseline before we add visualization, image-based segmentation, robustness testing, and YOLO.

## Step 4 visualization baseline

Step 4 creates annotated result images that show:

- object contour
- axis-aligned bounding box
- estimated center point
- recommended pick point
- estimated orientation axis
- center and angle text label

The output images are saved to:

```text
outputs/annotated/step4/
```

The preview grid is saved to:

```text
outputs/annotated/step4/annotation_grid.png
```

## Step 5 end-to-end demo

Step 5 adds a single demo command:

```powershell
py scripts\run_demo.py --regenerate
```

The command:

1. creates or loads the synthetic dataset
2. estimates pose from each mask
3. saves annotated result images
4. creates a visual preview grid
5. saves a JSON summary with center error, orientation error, and pose-estimation time

Generated demo outputs:

```text
outputs/annotated/demo/demo_grid.png
outputs/metrics/demo_summary.json
```

## Step 6 evaluation reports

Step 6 adds a proper evaluation command:

```powershell
py scripts\evaluate.py
```

Generated evaluation outputs:

```text
outputs/metrics/evaluation/per_sample_metrics.csv
outputs/metrics/evaluation/evaluation_summary.json
outputs/metrics/evaluation/evaluation_report.md
```

The report includes:

- mean, median, P90, and max center error
- mean, median, P90, and max orientation error
- inference-time statistics
- pass-rate checks against simple thresholds
- worst-case sample names
