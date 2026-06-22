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

1. Define scope and create repository foundation.
2. Build synthetic image generator with ground-truth labels.
3. Implement classical OpenCV pose estimation on synthetic objects.
4. Add visualization utilities.
5. Add first command-line demo.
6. Add evaluation metrics.
7. Add robustness transformations and robustness evaluation.
8. Add YOLO-based object detection path.
9. Integrate detection with pose estimation.
10. Build Streamlit GUI.
11. Add tests, examples, and failure-case analysis.
12. Polish GitHub repository and recruiter-facing README.
