"""Tests for visualization utilities."""

import cv2
import numpy as np

from pickpoint_vision.pose_estimation import estimate_pose_from_mask
from pickpoint_vision.visualization import annotate_pose_result


def test_annotate_pose_result_changes_image() -> None:
    """Annotation should preserve image shape and draw visible overlays."""
    image = np.full((240, 320, 3), 220, dtype=np.uint8)
    mask = np.zeros((240, 320), dtype=np.uint8)

    rect = ((160.0, 120.0), (120.0, 50.0), 35.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(mask, [box], contourIdx=-1, color=255, thickness=-1)
    cv2.drawContours(image, [box], contourIdx=-1, color=(80, 140, 180), thickness=-1)

    result = estimate_pose_from_mask(mask=mask, image_name="test.png")
    annotated = annotate_pose_result(image=image, mask=mask, result=result)

    assert annotated.shape == image.shape
    assert annotated.dtype == image.dtype
    assert np.mean(cv2.absdiff(annotated, image)) > 0.0
