# Step 14A: Interactive Contour Tuning

## Purpose

This step adds manual contour tuning controls to the Streamlit app.

The automatic pipeline is still the main demo path. Manual tuning is a debug feature that helps show how segmentation quality affects:

- binary mask
- contour
- center point
- orientation
- pick point

## New Streamlit controls

Enable:

```text
Interactive contour tuning -> Enable manual contour tuning
```

Available controls:

| Control | Purpose |
|---|---|
| Foreground sensitivity | Controls how easily pixels become foreground |
| Blur size | Smooths noise before thresholding |
| Open kernel | Removes small foreground speckles |
| Close kernel | Fills holes and gaps |
| Erode iterations | Shrinks the mask |
| Dilate iterations | Expands the mask |
| Minimum contour area | Filters tiny false contours |
| Contour smoothing | Smooths jagged object boundaries |
| Invert mask | Helps when background is selected instead of the object |

## Debug panels

The app shows:

1. Original image
2. Local background distance image
3. Tuned binary contour mask
4. Tuned contour + pick point

## Commands

Run the app:

```powershell
streamlit run app\streamlit_app.py
```

Run tests:

```powershell
py -m pytest tests\test_contour_tuning.py
py -m pytest
```

## Notes

Manual contour tuning improves segmentation and pose estimation. It does not train YOLO or change the YOLO detector itself.

This is useful for portfolio demos because it shows the full perception pipeline instead of only showing final predictions.
