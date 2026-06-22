"""Tests for image-based segmentation."""

import cv2
import numpy as np

from pickpoint_vision.pose_estimation import estimate_pose_from_mask
from pickpoint_vision.segmentation import foreground_coverage, segment_dark_foreground


def test_segment_dark_foreground_finds_object() -> None:
    """Dark foreground segmentation should recover a simple object."""
    image = np.full((240, 320, 3), 225, dtype=np.uint8)
    cv2.rectangle(image, (100, 80), (220, 160), (60, 90, 120), thickness=-1)

    mask = segment_dark_foreground(image)
    result = estimate_pose_from_mask(mask)

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert 5.0 < foreground_coverage(mask) < 20.0
    assert abs(result.center_x - 160.0) < 3.0
    assert abs(result.center_y - 120.0) < 3.0
