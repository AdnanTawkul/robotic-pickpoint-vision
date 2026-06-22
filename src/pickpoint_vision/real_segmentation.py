"""Real-image foreground segmentation helpers.

This module improves segmentation for normal phone images where objects are on a
tabletop or desk. It is designed for the integrated YOLO/OpenCV pipeline.

The key idea is local background subtraction:
- estimate the local background color from the border of the image/ROI
- segment pixels that are visually different from that border background
- clean the mask
- keep the best object-like connected component

This helps when YOLO finds a box but the old Otsu segmentation turns the whole
box into a rectangular mask, which gives the wrong orientation.
"""

from __future__ import annotations

import cv2
import numpy as np


def ensure_uint8_bgr(image: np.ndarray) -> np.ndarray:
    """Return image as uint8 BGR."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a grayscale or BGR image.")

    return image


def make_border_mask(
    image_height: int,
    image_width: int,
    border_ratio: float = 0.08,
) -> np.ndarray:
    """Create a boolean mask for the border ring of an image."""
    if image_height <= 0 or image_width <= 0:
        raise ValueError("Image dimensions must be positive.")

    border_width = max(3, int(round(min(image_height, image_width) * border_ratio)))
    border_width = min(border_width, max(1, image_height // 2), max(1, image_width // 2))

    mask = np.zeros((image_height, image_width), dtype=bool)
    mask[:border_width, :] = True
    mask[-border_width:, :] = True
    mask[:, :border_width] = True
    mask[:, -border_width:] = True

    return mask


def clean_binary_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Clean a binary mask with morphology."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    if kernel_size <= 1:
        return mask

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


def score_binary_mask(mask: np.ndarray) -> float:
    """Score how object-like a binary mask is."""
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    image_height, image_width = mask.shape[:2]
    image_area = float(image_height * image_width)

    foreground_count = int(np.count_nonzero(mask))
    if foreground_count <= 0:
        return -1.0

    area_fraction = foreground_count / image_area
    if area_fraction < 0.003 or area_fraction > 0.92:
        return -1.0

    border_mask = make_border_mask(image_height, image_width, border_ratio=0.06)
    border_foreground_fraction = float(np.count_nonzero(mask[border_mask])) / max(
        1.0,
        float(np.count_nonzero(border_mask)),
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1.0

    largest = max(contours, key=cv2.contourArea)
    largest_area = float(cv2.contourArea(largest))
    if largest_area <= 0.0:
        return -1.0

    dominance = largest_area / max(1.0, float(foreground_count))

    moments = cv2.moments(largest)
    if abs(moments["m00"]) < 1e-6:
        centrality = 0.0
    else:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        dx = abs(center_x - image_width / 2.0) / max(1.0, image_width / 2.0)
        dy = abs(center_y - image_height / 2.0) / max(1.0, image_height / 2.0)
        centrality = 1.0 - min(1.0, (dx + dy) / 2.0)

    border_penalty = 1.0 - min(1.0, border_foreground_fraction)
    area_balance = 1.0 - min(1.0, abs(area_fraction - 0.35))

    score = (
        0.40 * dominance
        + 0.25 * centrality
        + 0.20 * area_balance
        + 0.15 * border_penalty
    )

    if border_foreground_fraction > 0.70 and area_fraction > 0.60:
        score -= 0.45

    return float(score)


def keep_best_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep the best object-like connected component from a binary mask."""
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    best_score = -1.0
    best_component = np.zeros_like(mask)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 25:
            continue

        component = np.where(labels == label_id, 255, 0).astype(np.uint8)
        score = score_binary_mask(component)
        if score > best_score:
            best_score = score
            best_component = component

    if best_score < 0:
        return mask

    return best_component


def segment_by_local_background_contrast(
    image: np.ndarray,
    border_ratio: float = 0.08,
) -> np.ndarray:
    """Segment object pixels by contrast against the local border background."""
    image_bgr = ensure_uint8_bgr(image)
    image_height, image_width = image_bgr.shape[:2]

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_mask = make_border_mask(
        image_height=image_height,
        image_width=image_width,
        border_ratio=border_ratio,
    )

    background_color = np.median(lab[border_mask], axis=0)
    distance = np.linalg.norm(lab - background_color.reshape(1, 1, 3), axis=2)

    distance_u8 = cv2.normalize(
        distance,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    ).astype(np.uint8)
    distance_u8 = cv2.GaussianBlur(distance_u8, (5, 5), sigmaX=0)

    _, mask = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cleaned = clean_binary_mask(mask, kernel_size=5)
    return keep_best_connected_component(cleaned)


def _otsu_candidate_masks(image: np.ndarray) -> list[np.ndarray]:
    """Create classic Otsu polarity candidate masks."""
    image_bgr = ensure_uint8_bgr(image)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

    masks: list[np.ndarray] = []
    for threshold_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
        _, mask = cv2.threshold(gray, 0, 255, threshold_type + cv2.THRESH_OTSU)
        cleaned = clean_binary_mask(mask, kernel_size=5)
        masks.append(keep_best_connected_component(cleaned))

    return masks


def segment_real_object(image: np.ndarray) -> np.ndarray:
    """Segment the most object-like foreground region from a real image or ROI."""
    candidates: list[np.ndarray] = []

    try:
        candidates.append(segment_by_local_background_contrast(image))
    except Exception:
        pass

    candidates.extend(_otsu_candidate_masks(image))

    scored_candidates = [
        (candidate, score_binary_mask(candidate))
        for candidate in candidates
        if candidate is not None and np.count_nonzero(candidate) > 0
    ]

    if not scored_candidates:
        raise ValueError("Could not create any foreground mask candidates.")

    best_mask, best_score = max(scored_candidates, key=lambda item: item[1])
    if best_score < 0:
        raise ValueError("Could not create a usable object mask.")

    return best_mask
