"""Visualization utilities for pick-point estimation results."""

from __future__ import annotations

from pathlib import Path
import math

import cv2
import numpy as np

from pickpoint_vision.pose_estimation import PoseEstimationResult, find_largest_contour


DEFAULT_COLORS_BGR = {
    "contour": (0, 255, 255),
    "bbox": (255, 180, 0),
    "center": (0, 0, 255),
    "pick_point": (0, 255, 0),
    "orientation": (255, 0, 255),
    "text": (30, 30, 30),
    "text_background": (245, 245, 245),
}


def load_bgr_image(image_path: str | Path) -> np.ndarray:
    """Load an image as BGR."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def draw_label_background(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw readable text with a light background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = origin

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 4

    top_left = (x - padding, y - text_height - padding)
    bottom_right = (x + text_width + padding, y + baseline + padding)

    cv2.rectangle(
        image,
        top_left,
        bottom_right,
        DEFAULT_COLORS_BGR["text_background"],
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        origin,
        font,
        font_scale,
        DEFAULT_COLORS_BGR["text"],
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_orientation_axis(
    image: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
    length: float = 80.0,
    color: tuple[int, int, int] = DEFAULT_COLORS_BGR["orientation"],
    thickness: int = 3,
) -> None:
    """Draw the estimated orientation axis as a two-sided line."""
    center_x, center_y = center
    angle_rad = math.radians(angle_deg)

    dx = math.cos(angle_rad) * length / 2.0
    dy = math.sin(angle_rad) * length / 2.0

    point_a = (int(round(center_x - dx)), int(round(center_y - dy)))
    point_b = (int(round(center_x + dx)), int(round(center_y + dy)))

    cv2.line(image, point_a, point_b, color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.circle(image, point_b, radius=4, color=color, thickness=-1)


def annotate_pose_result(
    image: np.ndarray,
    mask: np.ndarray,
    result: PoseEstimationResult,
    orientation_source: str = "pca",
) -> np.ndarray:
    """Draw contour, bounding box, center point, pick point, and orientation axis."""
    annotated = image.copy()

    contour = find_largest_contour(mask)

    cv2.drawContours(
        annotated,
        [contour],
        contourIdx=-1,
        color=DEFAULT_COLORS_BGR["contour"],
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    cv2.rectangle(
        annotated,
        (result.bbox_x, result.bbox_y),
        (result.bbox_x + result.bbox_width, result.bbox_y + result.bbox_height),
        DEFAULT_COLORS_BGR["bbox"],
        thickness=2,
    )

    center = (result.center_x, result.center_y)
    pick_point = (result.pick_x, result.pick_y)

    if orientation_source == "min_area_rect":
        angle_deg = result.angle_deg_min_area_rect
    else:
        angle_deg = result.angle_deg_pca

    axis_length = max(50.0, min(result.bbox_width, result.bbox_height) * 1.4)
    draw_orientation_axis(
        annotated,
        center=center,
        angle_deg=angle_deg,
        length=axis_length,
    )

    cv2.circle(
        annotated,
        (int(round(center[0])), int(round(center[1]))),
        radius=6,
        color=DEFAULT_COLORS_BGR["center"],
        thickness=-1,
    )
    cv2.circle(
        annotated,
        (int(round(pick_point[0])), int(round(pick_point[1]))),
        radius=10,
        color=DEFAULT_COLORS_BGR["pick_point"],
        thickness=2,
    )

    label = (
        f"center=({result.center_x:.1f}, {result.center_y:.1f})  "
        f"angle={angle_deg:.1f} deg"
    )
    text_x = max(10, result.bbox_x)
    text_y = max(25, result.bbox_y - 8)
    draw_label_background(annotated, label, (text_x, text_y))

    return annotated


def annotate_pose_from_files(
    image_path: str | Path,
    mask_path: str | Path,
    result: PoseEstimationResult,
    output_path: str | Path,
    orientation_source: str = "pca",
) -> Path:
    """Load image and mask, draw annotation, and save result."""
    image = load_bgr_image(image_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    annotated = annotate_pose_result(
        image=image,
        mask=mask,
        result=result,
        orientation_source=orientation_source,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    return output_path


def create_annotation_grid(
    image_paths: list[Path],
    output_path: str | Path,
    max_images: int = 12,
    cell_width: int = 320,
    cell_height: int = 240,
) -> Path:
    """Create a preview grid from annotated images."""
    selected_paths = image_paths[:max_images]
    if not selected_paths:
        raise ValueError("No images provided for annotation grid.")

    images: list[np.ndarray] = []
    for image_path in selected_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        resized = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        images.append(resized)

    if not images:
        raise ValueError("Could not read any images for annotation grid.")

    columns = 3
    rows = int(math.ceil(len(images) / columns))
    grid = np.full((rows * cell_height, columns * cell_width, 3), 245, dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // columns
        column = idx % columns
        y0 = row * cell_height
        x0 = column * cell_width
        grid[y0 : y0 + cell_height, x0 : x0 + cell_width] = image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)

    return output_path
