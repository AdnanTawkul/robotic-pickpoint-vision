# Failure-Case Analysis

This document summarizes known limitations and practical failure modes.

## Why failure analysis matters

A robotics perception project is more credible when it shows where the pipeline works and where it fails. This project intentionally includes robustness tests and real-image examples to document limitations rather than hiding them.

---

## 1. YOLO misses unusual objects

### Observation

Off-the-shelf YOLO may miss objects that are not common COCO-style categories or are photographed from unusual views.

Examples observed during testing:

- bottle image: no YOLO detection
- toy car image: no YOLO detection
- cup image: no YOLO detection
- mouse image: YOLO behavior depends on view and confidence
- scissors/tweezers-like tools: more likely to be detected

### Cause

The default YOLO model is trained on general object categories. It is not specialized for arbitrary industrial pick objects.

### Impact

The system may need OpenCV fallback segmentation or a custom-trained detector for the target domain.

### Possible fixes

- train a custom YOLO detector on pick-object images
- use YOLO segmentation models instead of bounding-box detection
- add a custom dataset of industrial parts
- use segmentation-first methods for unknown objects

---

## 2. OpenCV fallback can include background

### Observation

When an object and background have similar intensity or color, full-image OpenCV fallback can segment parts of the background.

### Cause

The fallback uses classical thresholding and morphology. This is simple and fast, but not semantic.

### Impact

The estimated contour may include background texture, causing incorrect center and orientation.

### Possible fixes

- improve preprocessing
- add background normalization
- use color-space thresholding
- restrict segmentation to user-selected or detected regions
- use learned segmentation masks

---

## 3. Dark objects on dark backgrounds

### Observation

Dark objects on a black or dark textured surface can produce noisy contours.

### Cause

Foreground/background contrast is weak.

### Impact

The object contour may be fragmented or mixed with background texture.

### Possible fixes

- improve lighting
- use a contrasting surface
- add adaptive thresholding
- use edge-aware segmentation
- train a custom model

---

## 4. Partial occlusion

### Observation

Robustness evaluation showed that partial occlusion can create very large center and orientation errors.

### Cause

The visible contour no longer represents the full object geometry.

### Impact

The estimated center and orientation can shift toward the visible region.

### Possible fixes

- detect occlusion and reject low-confidence estimates
- use object-shape priors
- use instance segmentation
- use multiple camera views
- use temporal tracking across frames

---

## 5. Symmetric objects have ambiguous orientation

### Observation

Round cups, bowls, circular caps, and near-symmetric shapes do not have a meaningful unique orientation.

### Cause

The principal axis may be unstable when the object has rotational symmetry.

### Impact

Orientation estimates can change even when the center is correct.

### Possible fixes

- report orientation confidence
- suppress orientation for circular objects
- use object-specific orientation rules
- estimate handle direction for cups/bowls when visible

---

## 6. Thin objects and holes

### Observation

Thin tools or objects with holes can produce contours where the geometric center differs from a good grasp point.

### Cause

Contour moments and PCA describe visible 2D shape, not grasp stability.

### Impact

The pick point may be visually centered but not mechanically ideal.

### Possible fixes

- add object-specific pick-point rules
- use skeletonization for thin tools
- estimate graspable regions instead of only center
- train a grasp-point model

---

## Current conclusion

The pipeline is a strong portfolio demonstration because it includes:

- successful synthetic evaluation
- real-image demos
- robustness testing
- documented failure cases
- a clear path toward production improvements

It is not presented as a production robotic grasping system. It is a practical perception prototype.
