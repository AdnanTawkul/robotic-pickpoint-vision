"""Tests for real-image segmentation helpers."""

import cv2
import numpy as np

from pickpoint_vision.pose_estimation import axis_orientation_error_deg, estimate_pose_from_mask
from pickpoint_vision.real_segmentation import (
    score_binary_mask,
    segment_by_local_background_contrast,
    segment_real_object,
)


def test_segment_real_object_light_rotated_object_on_dark_background() -> None:
    """Local background segmentation should recover a light rotated object."""
    image = np.full((300, 400, 3), (35, 35, 35), dtype=np.uint8)
    rect = ((210.0, 150.0), (180.0, 55.0), 32.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(image, [box], contourIdx=-1, color=(220, 220, 210), thickness=-1)

    mask = segment_real_object(image)
    pose = estimate_pose_from_mask(mask)

    assert abs(pose.center_x - 210.0) < 5.0
    assert abs(pose.center_y - 150.0) < 5.0
    assert axis_orientation_error_deg(pose.angle_deg_pca, 32.0) < 8.0


def test_local_background_segmentation_scores_better_than_full_mask() -> None:
    """A real object mask should score better than a full rectangular mask."""
    image = np.full((220, 320, 3), (30, 30, 30), dtype=np.uint8)
    cv2.ellipse(image, (160, 110), (95, 30), 35, 0, 360, (230, 230, 220), thickness=-1)

    object_mask = segment_by_local_background_contrast(image)
    full_mask = np.full(object_mask.shape, 255, dtype=np.uint8)

    assert score_binary_mask(object_mask) > score_binary_mask(full_mask)
    assert np.count_nonzero(object_mask) < np.count_nonzero(full_mask) * 0.5
