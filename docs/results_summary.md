# Results Summary

## Clean synthetic baseline

The clean baseline evaluates pose estimation from known synthetic masks.

| Metric | Result |
|---|---:|
| Mean center error | 0.334 px |
| Median center error | 0.052 px |
| P90 center error | 0.715 px |
| Max center error | 0.805 px |
| Mean PCA orientation error | 0.832 deg |
| P90 PCA orientation error | 2.166 deg |
| Mean minAreaRect orientation error | 2.776 deg |
| Center error <= 2 px | 100% |
| PCA orientation error <= 5 deg | 100% |

## Robustness summary

The image-based robustness evaluation uses degraded RGB images and OpenCV segmentation.

| Degradation | Expected behavior |
|---|---|
| Mild blur | Stable |
| Strong blur | Usually stable for large objects |
| Mild noise | Stable after preprocessing |
| Strong noise | Usually stable, but may create artifacts |
| Brightness changes | Stable when foreground/background contrast remains |
| Contrast changes | Stable unless contrast becomes too low |
| Partial occlusion | Large errors expected |

## Real-image notes

The integrated pipeline works on normal phone images but quality depends on object/background contrast and YOLO detections.

Observed examples:

- YOLO detected scissors/tweezers-like objects more reliably than unusual objects.
- YOLO missed some objects such as the bottle, toy car, cup, and mouse.
- OpenCV fallback still produced pick-point estimates for every sample image.
- Dark objects on dark backgrounds can create noisy contours.
- Symmetric objects can have ambiguous orientation.

## Interpretation

The project is not claiming production-grade robotic grasping. It demonstrates a practical and measurable perception pipeline with clear strengths, limitations, and next-step improvements.
