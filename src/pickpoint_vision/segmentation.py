"""Simple image-based segmentation for synthetic pick-point scenes.

This is intentionally classical OpenCV segmentation:
- grayscale conversion
- optional blur
- Otsu thresholding
- morphology cleanup

The synthetic dataset uses dark objects on a light background, so an inverted
threshold is a good first baseline. This gives us an image-based path for
robustness evaluation before adding YOLO.
"""

from __future__ import annotations

import cv2
import numpy as np


def segment_dark_foreground(
    image: np.ndarray,
    blur_kernel_size: int = 5,
    morphology_kernel_size: int = 5,
) -> np.ndarray:
    """Segment dark foreground objects from a light background.

    Returns a binary uint8 mask with foreground as 255.
    """
    if image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image.copy()

    if grayscale.dtype != np.uint8:
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)

    if blur_kernel_size > 1:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        grayscale = cv2.GaussianBlur(
            grayscale,
            (blur_kernel_size, blur_kernel_size),
            sigmaX=0,
        )

    _, mask = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    if morphology_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphology_kernel_size, morphology_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def foreground_coverage(mask: np.ndarray) -> float:
    """Return the foreground percentage in a binary mask."""
    if mask.size == 0:
        raise ValueError("Cannot compute coverage for an empty mask.")

    foreground_pixels = int(np.count_nonzero(mask))
    return float((foreground_pixels / mask.size) * 100.0)
