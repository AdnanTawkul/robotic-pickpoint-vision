"""Interactive contour tuning helpers.

These utilities are used by the Streamlit demo to let users manually tune the
segmentation mask and contour used for pick-point estimation.

This does not replace the automatic pipeline. It adds a debug/tuning path that
is useful when real-image segmentation is imperfect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from pickpoint_vision.pose_estimation import PoseEstimationResult, estimate_pose_from_mask
from pickpoint_vision.real_segmentation import ensure_uint8_bgr, make_border_mask


@dataclass(frozen=True)
class ContourTuningConfig:
    """Configuration for manual contour tuning."""

    foreground_sensitivity: float = 0.50
    blur_kernel_size: int = 5
    open_kernel_size: int = 5
    close_kernel_size: int = 7
    erode_iterations: int = 0
    dilate_iterations: int = 0
    min_contour_area: int = 250
    contour_smoothing: float = 0.005
    invert_mask: bool = False

    def to_dict(self) -> dict[str, object]:
        """Convert config to a serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ContourTuningResult:
    """Result returned by the manual contour tuning path."""

    mask: np.ndarray
    contour_mask: np.ndarray
    pose_result: PoseEstimationResult
    contour_area: float
    config: ContourTuningConfig


def odd_kernel_size(value: int) -> int:
    """Return a valid odd OpenCV kernel size."""
    value = int(value)
    if value <= 1:
        return 1
    if value % 2 == 0:
        value += 1
    return value


def apply_optional_blur(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur if the kernel size is greater than 1."""
    kernel_size = odd_kernel_size(kernel_size)
    if kernel_size <= 1:
        return gray
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX=0)


def apply_morphology(mask: np.ndarray, config: ContourTuningConfig) -> np.ndarray:
    """Apply open, close, erode, and dilate operations."""
    cleaned = np.where(mask > 0, 255, 0).astype(np.uint8)

    open_size = odd_kernel_size(config.open_kernel_size)
    if open_size > 1:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    close_size = odd_kernel_size(config.close_kernel_size)
    if close_size > 1:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    if config.erode_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.erode(cleaned, kernel, iterations=int(config.erode_iterations))

    if config.dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.dilate(cleaned, kernel, iterations=int(config.dilate_iterations))

    return cleaned


def create_local_background_distance_image(image: np.ndarray) -> np.ndarray:
    """Create a grayscale distance image from local border background color."""
    image_bgr = ensure_uint8_bgr(image)
    image_height, image_width = image_bgr.shape[:2]

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = make_border_mask(
        image_height=image_height,
        image_width=image_width,
        border_ratio=0.08,
    )

    background_color = np.median(lab[border], axis=0)
    distance = np.linalg.norm(lab - background_color.reshape(1, 1, 3), axis=2)

    distance_u8 = cv2.normalize(
        distance,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    ).astype(np.uint8)

    return distance_u8


def threshold_distance_image(
    distance_image: np.ndarray,
    foreground_sensitivity: float,
    invert_mask: bool = False,
) -> np.ndarray:
    """Threshold the distance image into a binary foreground mask.

    Sensitivity behavior:
    - lower sensitivity -> stricter foreground threshold
    - higher sensitivity -> more pixels become foreground
    """
    sensitivity = float(np.clip(foreground_sensitivity, 0.0, 1.0))
    blurred = apply_optional_blur(distance_image, kernel_size=5)

    otsu_value, _ = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Sensitivity controls the threshold around Otsu.
    # high sensitivity lowers the threshold; low sensitivity raises it.
    threshold_scale = 1.65 - sensitivity
    threshold_value = int(np.clip(otsu_value * threshold_scale, 3, 252))

    _, mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    if invert_mask:
        mask = cv2.bitwise_not(mask)

    return mask


def keep_best_tuned_contour(mask: np.ndarray, config: ContourTuningConfig) -> np.ndarray:
    """Keep the best contour after area filtering and optional smoothing."""
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(1.0, float(config.min_contour_area))

    valid_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= min_area
    ]

    if not valid_contours:
        return np.zeros_like(mask)

    best_contour = max(valid_contours, key=cv2.contourArea)

    epsilon_fraction = float(np.clip(config.contour_smoothing, 0.0, 0.05))
    if epsilon_fraction > 0.0:
        perimeter = cv2.arcLength(best_contour, closed=True)
        epsilon = epsilon_fraction * perimeter
        best_contour = cv2.approxPolyDP(best_contour, epsilon, closed=True)

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [best_contour], contourIdx=-1, color=255, thickness=-1)

    return contour_mask


def segment_with_contour_tuning(
    image: np.ndarray,
    config: ContourTuningConfig,
) -> np.ndarray:
    """Segment an object mask using manual contour tuning settings."""
    distance_image = create_local_background_distance_image(image)
    distance_image = apply_optional_blur(distance_image, config.blur_kernel_size)

    mask = threshold_distance_image(
        distance_image=distance_image,
        foreground_sensitivity=config.foreground_sensitivity,
        invert_mask=config.invert_mask,
    )
    mask = apply_morphology(mask, config)
    contour_mask = keep_best_tuned_contour(mask, config)

    return contour_mask


def estimate_pose_with_contour_tuning(
    image: np.ndarray,
    config: ContourTuningConfig,
    image_name: str = "",
) -> ContourTuningResult:
    """Segment the object with tuning controls and estimate pose."""
    mask = segment_with_contour_tuning(image=image, config=config)
    if np.count_nonzero(mask) == 0:
        raise ValueError("Manual contour tuning produced an empty mask.")

    pose_result = estimate_pose_from_mask(mask, image_name=image_name)
    return ContourTuningResult(
        mask=mask,
        contour_mask=mask,
        pose_result=pose_result,
        contour_area=pose_result.contour_area,
        config=config,
    )
