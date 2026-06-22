"""Tests for OpenCV pose estimation."""

import cv2
import numpy as np

from pickpoint_vision.pose_estimation import (
    axis_orientation_error_deg,
    estimate_pose_from_mask,
    normalize_axis_angle_deg,
)


def create_rotated_rectangle_mask(
    width: int = 320,
    height: int = 240,
    center: tuple[float, float] = (160.0, 120.0),
    size: tuple[float, float] = (120.0, 50.0),
    angle_deg: float = 35.0,
) -> np.ndarray:
    """Create a simple binary mask containing one rotated rectangle."""
    mask = np.zeros((height, width), dtype=np.uint8)
    rect = (center, size, angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(mask, [box], contourIdx=-1, color=255, thickness=-1)
    return mask


def test_normalize_axis_angle_deg() -> None:
    """Axis-angle normalization should treat 180-degree flips as equivalent."""
    assert normalize_axis_angle_deg(0.0) == 0.0
    assert normalize_axis_angle_deg(180.0) == 0.0
    assert normalize_axis_angle_deg(90.0) == -90.0
    assert normalize_axis_angle_deg(-100.0) == 80.0


def test_axis_orientation_error_deg() -> None:
    """Orientation error should respect 180-degree axis symmetry."""
    assert axis_orientation_error_deg(10.0, 10.0) == 0.0
    assert axis_orientation_error_deg(10.0, 190.0) == 0.0
    assert axis_orientation_error_deg(85.0, -85.0) == 10.0


def test_estimate_pose_from_mask_rotated_rectangle() -> None:
    """Pose estimation should recover center and orientation from a clean rectangle mask."""
    expected_center = (160.0, 120.0)
    expected_angle = 35.0

    mask = create_rotated_rectangle_mask(center=expected_center, angle_deg=expected_angle)
    result = estimate_pose_from_mask(mask, image_name="test.png")

    center_error = ((result.center_x - expected_center[0]) ** 2 + (result.center_y - expected_center[1]) ** 2) ** 0.5
    angle_error = axis_orientation_error_deg(result.angle_deg_pca, expected_angle)

    assert result.image_name == "test.png"
    assert center_error < 1.0
    assert angle_error < 2.0
    assert result.contour_area > 0.0
    assert result.bbox_width > 0
    assert result.bbox_height > 0
