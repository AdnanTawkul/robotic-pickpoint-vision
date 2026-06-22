"""2D pose estimation from binary object masks.

This module estimates:
- object center from contour moments
- object orientation from PCA
- object orientation from OpenCV minAreaRect
- pick point, currently equal to the estimated center

The first implementation uses masks from the synthetic dataset. This gives a clean,
controlled baseline before we later add image-based segmentation and YOLO detection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class PoseEstimationResult:
    """Estimated 2D object pose from one binary mask."""

    image_name: str
    center_x: float
    center_y: float
    pick_x: float
    pick_y: float
    angle_deg_pca: float
    angle_deg_min_area_rect: float
    contour_area: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int

    def to_dict(self) -> dict[str, object]:
        """Convert result to a serializable dictionary."""
        return asdict(self)


def normalize_axis_angle_deg(angle_deg: float) -> float:
    """Normalize an undirected object-axis angle to [-90, 90).

    Object orientation in this project describes an axis, not an arrow.
    Therefore, 0 and 180 degrees represent the same orientation.
    """
    return float(((angle_deg + 90.0) % 180.0) - 90.0)


def axis_orientation_error_deg(predicted_deg: float, target_deg: float) -> float:
    """Return the smallest absolute angle error for an undirected axis."""
    difference = normalize_axis_angle_deg(predicted_deg - target_deg)
    return abs(float(difference))


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Load a mask image and return a binary uint8 mask with values 0 or 255."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    return preprocess_binary_mask(mask)


def preprocess_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a mask-like image into a clean binary mask."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    return binary_mask


def find_largest_contour(mask: np.ndarray, min_area: float = 25.0) -> np.ndarray:
    """Find the largest external contour in a binary mask."""
    binary_mask = preprocess_binary_mask(mask)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("No contours found in mask.")

    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))

    if area < min_area:
        raise ValueError(f"Largest contour area is too small: {area:.2f}")

    return largest_contour


def estimate_center_from_contour(contour: np.ndarray) -> tuple[float, float]:
    """Estimate object center using image moments."""
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        raise ValueError("Contour has near-zero area; cannot estimate center.")

    center_x = float(moments["m10"] / moments["m00"])
    center_y = float(moments["m01"] / moments["m00"])
    return center_x, center_y


def estimate_orientation_pca(contour: np.ndarray) -> float:
    """Estimate object orientation using PCA over contour points."""
    points = contour.reshape(-1, 2).astype(np.float64)

    if len(points) < 5:
        raise ValueError("At least 5 contour points are required for PCA orientation.")

    mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=2)
    del mean

    principal_axis = eigenvectors[0]
    angle_rad = math.atan2(float(principal_axis[1]), float(principal_axis[0]))
    angle_deg = math.degrees(angle_rad)

    return normalize_axis_angle_deg(angle_deg)


def estimate_orientation_min_area_rect(contour: np.ndarray) -> float:
    """Estimate object orientation using OpenCV minAreaRect.

    OpenCV's angle convention depends on which rectangle side is considered width.
    We convert it to the orientation of the longer object axis.
    """
    (_, _), (width, height), angle_deg = cv2.minAreaRect(contour)

    if width < height:
        angle_deg += 90.0

    return normalize_axis_angle_deg(float(angle_deg))


def estimate_pose_from_mask(
    mask: np.ndarray,
    image_name: str = "",
) -> PoseEstimationResult:
    """Estimate object center, orientation, and pick point from a binary mask."""
    contour = find_largest_contour(mask)
    center_x, center_y = estimate_center_from_contour(contour)
    angle_deg_pca = estimate_orientation_pca(contour)
    angle_deg_min_area_rect = estimate_orientation_min_area_rect(contour)
    bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(contour)
    contour_area = float(cv2.contourArea(contour))

    return PoseEstimationResult(
        image_name=image_name,
        center_x=round(center_x, 3),
        center_y=round(center_y, 3),
        pick_x=round(center_x, 3),
        pick_y=round(center_y, 3),
        angle_deg_pca=round(angle_deg_pca, 3),
        angle_deg_min_area_rect=round(angle_deg_min_area_rect, 3),
        contour_area=round(contour_area, 3),
        bbox_x=int(bbox_x),
        bbox_y=int(bbox_y),
        bbox_width=int(bbox_width),
        bbox_height=int(bbox_height),
    )


def estimate_pose_from_mask_file(mask_path: str | Path, image_name: str = "") -> PoseEstimationResult:
    """Load a mask file and estimate object pose."""
    mask_path = Path(mask_path)
    mask = load_binary_mask(mask_path)

    if not image_name:
        image_name = mask_path.name.replace("_mask", "")

    return estimate_pose_from_mask(mask=mask, image_name=image_name)


def save_pose_results_csv(
    results: list[PoseEstimationResult],
    output_path: str | Path,
) -> None:
    """Save pose-estimation results to CSV."""
    if not results:
        raise ValueError("Cannot save an empty pose result list.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].to_dict().keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())
