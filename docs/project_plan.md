# Project Plan

## Project title

Vision-Based Pick-Point Estimation for Robotic Manipulation

## Scope

This project estimates 2D pick points and object orientations from RGB images without requiring a robot arm, depth camera, or external equipment.

## Final status

All planned steps are complete.

## Completed roadmap

1. Define scope and create repository foundation. **Done**
2. Build synthetic image generator with ground-truth labels. **Done**
3. Implement classical OpenCV pose estimation on synthetic objects. **Done**
4. Add visualization utilities. **Done**
5. Add first command-line demo. **Done**
6. Add evaluation metrics. **Done**
7. Add robustness transformations. **Done**
8. Add robustness evaluation. **Done**
9. Add YOLO-based object detection path. **Done**
10. Integrate detection with pose estimation. **Done**
11. Build Streamlit GUI. **Done**
12. Polish GitHub repository and recruiter-facing README. **Done**

## Final demo behavior

The final demo supports:

- upload image
- choose YOLO + OpenCV fallback or OpenCV-only mode
- estimate object center
- estimate orientation
- visualize contour, bounding box, center, pick point, and orientation axis
- download annotated image
- download result CSV

## Final repository strengths

- practical robotics perception framing
- measurable evaluation metrics
- robustness and failure-case analysis
- real-image demo path
- clean Python package layout
- unit tests
- professional documentation
