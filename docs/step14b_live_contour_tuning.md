# Step 14B: Live Contour Tuning Workflow

## Purpose

Step 14A added contour tuning sliders, but the workflow still felt like a separate regeneration step.

Step 14B reorganizes the Streamlit app into two tabs:

1. Automatic pipeline
2. Live contour tuning

The live contour tuning tab updates immediately whenever sliders move. It reruns only the lightweight OpenCV mask/contour/pose step, not the full YOLO pipeline.

## What is real-time?

Streamlit reruns the script whenever a widget changes. Since the live contour sliders are not inside a form, each slider movement recomputes:

- local background distance image
- binary mask
- best contour
- center point
- orientation
- pick point
- downloadable result image
- downloadable result CSV

## What this improves

This makes the tuning workflow much faster:

- upload image once
- enable live contour tuning
- move sliders
- immediately see the mask and contour update
- use the tuned result as the final contour-based estimate

## Commands

Run:

```powershell
streamlit run app\streamlit_app.py
```

Then:

1. upload an image
2. open the `Live contour tuning` tab
3. enable `Live contour tuning` in the sidebar
4. move the sliders
5. download the tuned annotated image or CSV

## Notes

This still does not change the YOLO detector. It improves the contour/segmentation and reruns the pose estimation from the tuned mask.
