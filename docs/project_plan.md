# Project Plan

## Project title

Vision-Based Pick-Point Estimation for Robotic Manipulation

## Scope

This project estimates 2D pick points and object orientations from RGB images without requiring a robot arm, depth camera, or external equipment.

## Final demo

The final demo allows a user to upload an image, run the perception pipeline, and view:

- detected object bounding boxes, when YOLO is enabled
- object contours or masks
- center point
- estimated object orientation
- recommended pick point
- inference summary
- saved/downloadable annotated output image

## Roadmap

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
11. Build Streamlit GUI. **Current**
12. Polish GitHub repository and recruiter-facing README.

## Step 11 Streamlit GUI

Step 11 adds an interactive demo app.

The command is:

```powershell
streamlit run app\streamlit_app.py
```

The app supports:

- image upload
- YOLO + OpenCV fallback mode
- OpenCV-only mode
- YOLO confidence threshold
- optional YOLO class filter
- pick-point and orientation visualization
- metrics table
- annotated image download
- result CSV download

Generated app outputs:

```text
outputs/streamlit/uploads/
outputs/streamlit/annotated/
```
