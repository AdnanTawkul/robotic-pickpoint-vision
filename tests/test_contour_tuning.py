"""Tests for interactive contour tuning helpers."""

import cv2
import numpy as np

from pickpoint_vision.contour_tuning import (
    ContourTuningConfig,
    estimate_pose_with_contour_tuning,
    odd_kernel_size,
    segment_with_contour_tuning,
)
from pickpoint_vision.pose_estimation import axis_orientation_error_deg


def test_odd_kernel_size() -> None:
    """Kernel sizes should be valid odd values."""
    assert odd_kernel_size(1) == 1
    assert odd_kernel_size(2) == 3
    assert odd_kernel_size(6) == 7


def test_segment_with_contour_tuning_recovers_light_object() -> None:
    """Manual tuning should recover a light rotated object on a dark background."""
    image = np.full((260, 360, 3), (30, 30, 30), dtype=np.uint8)
    rect = ((180.0, 130.0), (145.0, 42.0), 28.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(image, [box], contourIdx=-1, color=(230, 225, 215), thickness=-1)

    config = ContourTuningConfig(
        foreground_sensitivity=0.55,
        blur_kernel_size=5,
        open_kernel_size=3,
        close_kernel_size=5,
        min_contour_area=100,
    )
    mask = segment_with_contour_tuning(image=image, config=config)

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 1000


def test_estimate_pose_with_contour_tuning() -> None:
    """Manual tuning should produce a usable pose estimate."""
    image = np.full((260, 360, 3), (30, 30, 30), dtype=np.uint8)
    rect = ((180.0, 130.0), (145.0, 42.0), 28.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(image, [box], contourIdx=-1, color=(230, 225, 215), thickness=-1)

    config = ContourTuningConfig(
        foreground_sensitivity=0.55,
        blur_kernel_size=5,
        open_kernel_size=3,
        close_kernel_size=5,
        min_contour_area=100,
    )
    result = estimate_pose_with_contour_tuning(
        image=image,
        config=config,
        image_name="test.png",
    )

    assert abs(result.pose_result.center_x - 180.0) < 5.0
    assert abs(result.pose_result.center_y - 130.0) < 5.0
    assert axis_orientation_error_deg(result.pose_result.angle_deg_pca, 28.0) < 10.0
