"""Synthetic dataset generation for pick-point estimation.

The generated data is intentionally simple at first:
- one object per image
- clean background with mild texture/noise
- known object center
- known object orientation
- known pick point, currently equal to the object center

Later steps will use this ground truth for pose-estimation evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math
import random

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticObjectLabel:
    """Ground-truth label for one synthetic object."""

    image_name: str
    image_path: str
    object_id: int
    shape: str
    center_x: float
    center_y: float
    width: float
    height: float
    angle_deg: float
    pick_x: float
    pick_y: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    image_width: int
    image_height: int

    def to_dict(self) -> dict[str, object]:
        """Convert label to a serializable dictionary."""
        return asdict(self)


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize an angle to the range [-90, 90)."""
    normalized = ((angle_deg + 90.0) % 180.0) - 90.0
    return float(normalized)


def _create_background(
    image_width: int,
    image_height: int,
    rng: random.Random,
) -> np.ndarray:
    """Create a light textured background."""
    base_value = rng.randint(205, 238)
    background = np.full((image_height, image_width, 3), base_value, dtype=np.uint8)

    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(
        loc=0.0,
        scale=4.0,
        size=background.shape,
    )
    noisy_background = background.astype(np.float32) + noise
    return np.clip(noisy_background, 0, 255).astype(np.uint8)


def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return x, y, width, height for the non-zero region of a mask."""
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        return 0, 0, 0, 0

    x, y, width, height = cv2.boundingRect(nonzero)
    return int(x), int(y), int(width), int(height)


def _draw_rotated_rectangle(
    image: np.ndarray,
    mask: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    angle_deg: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled rotated rectangle on the image and mask."""
    rect = (center, size, angle_deg)
    box_points = cv2.boxPoints(rect).astype(np.int32)

    cv2.drawContours(image, [box_points], contourIdx=-1, color=color, thickness=-1)
    cv2.drawContours(mask, [box_points], contourIdx=-1, color=255, thickness=-1)

    edge_color = tuple(max(0, int(channel) - 45) for channel in color)
    cv2.polylines(image, [box_points], isClosed=True, color=edge_color, thickness=2)


def _draw_rotated_ellipse(
    image: np.ndarray,
    mask: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    angle_deg: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled rotated ellipse on the image and mask."""
    axes = (max(1, int(size[0] / 2)), max(1, int(size[1] / 2)))
    center_int = (int(round(center[0])), int(round(center[1])))

    cv2.ellipse(image, center_int, axes, angle_deg, 0, 360, color, thickness=-1)
    cv2.ellipse(mask, center_int, axes, angle_deg, 0, 360, 255, thickness=-1)

    edge_color = tuple(max(0, int(channel) - 45) for channel in color)
    cv2.ellipse(image, center_int, axes, angle_deg, 0, 360, edge_color, thickness=2)


def generate_synthetic_sample(
    index: int,
    image_width: int = 640,
    image_height: int = 480,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray, SyntheticObjectLabel]:
    """Generate one synthetic image, binary mask, and label."""
    if rng is None:
        rng = random.Random()

    image = _create_background(image_width=image_width, image_height=image_height, rng=rng)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    shape = rng.choice(["rectangle", "ellipse"])

    width = float(rng.randint(90, 210))
    height = float(rng.randint(35, 110))

    if height > width:
        width, height = height, width

    margin = int(math.ceil(max(width, height) / 2.0)) + 20
    center_x = float(rng.randint(margin, image_width - margin))
    center_y = float(rng.randint(margin, image_height - margin))
    angle_deg = normalize_angle_deg(rng.uniform(-85.0, 85.0))

    color = (
        rng.randint(40, 190),
        rng.randint(40, 190),
        rng.randint(40, 190),
    )

    if shape == "rectangle":
        _draw_rotated_rectangle(
            image=image,
            mask=mask,
            center=(center_x, center_y),
            size=(width, height),
            angle_deg=angle_deg,
            color=color,
        )
    else:
        _draw_rotated_ellipse(
            image=image,
            mask=mask,
            center=(center_x, center_y),
            size=(width, height),
            angle_deg=angle_deg,
            color=color,
        )

    bbox_x, bbox_y, bbox_width, bbox_height = _mask_to_bbox(mask)
    image_name = f"synthetic_{index:04d}.png"
    relative_image_path = f"images/{image_name}"

    label = SyntheticObjectLabel(
        image_name=image_name,
        image_path=relative_image_path,
        object_id=0,
        shape=shape,
        center_x=round(center_x, 3),
        center_y=round(center_y, 3),
        width=round(width, 3),
        height=round(height, 3),
        angle_deg=round(angle_deg, 3),
        pick_x=round(center_x, 3),
        pick_y=round(center_y, 3),
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        image_width=image_width,
        image_height=image_height,
    )

    return image, mask, label


def save_labels_csv(labels: list[SyntheticObjectLabel], output_path: str | Path) -> None:
    """Save synthetic labels to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not labels:
        raise ValueError("Cannot save an empty label list.")

    fieldnames = list(labels[0].to_dict().keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for label in labels:
            writer.writerow(label.to_dict())


def save_labels_json(labels: list[SyntheticObjectLabel], output_path: str | Path) -> None:
    """Save synthetic labels to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump([label.to_dict() for label in labels], file, indent=2)


def create_preview_grid(
    image_paths: list[Path],
    output_path: str | Path,
    max_images: int = 12,
    cell_width: int = 320,
    cell_height: int = 240,
) -> None:
    """Create a simple preview grid from generated images."""
    selected_paths = image_paths[:max_images]
    if not selected_paths:
        return

    images: list[np.ndarray] = []
    for image_path in selected_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        resized = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        images.append(resized)

    if not images:
        return

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


def create_synthetic_dataset(
    output_dir: str | Path,
    num_images: int,
    image_width: int = 640,
    image_height: int = 480,
    seed: int = 42,
    clear_existing: bool = False,
) -> list[SyntheticObjectLabel]:
    """Create a synthetic dataset with images, masks, and labels.

    Directory layout:

        output_dir/
        ├── images/
        ├── masks/
        ├── labels.csv
        ├── labels.json
        └── preview_grid.png
    """
    if num_images <= 0:
        raise ValueError("num_images must be greater than zero.")

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"

    if clear_existing and output_dir.exists():
        for subdir in [images_dir, masks_dir]:
            if subdir.exists():
                for file_path in subdir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    labels: list[SyntheticObjectLabel] = []
    image_paths: list[Path] = []

    for index in range(num_images):
        image, mask, label = generate_synthetic_sample(
            index=index,
            image_width=image_width,
            image_height=image_height,
            rng=rng,
        )

        image_path = images_dir / label.image_name
        mask_path = masks_dir / label.image_name.replace(".png", "_mask.png")

        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), mask)

        labels.append(label)
        image_paths.append(image_path)

    save_labels_csv(labels, output_dir / "labels.csv")
    save_labels_json(labels, output_dir / "labels.json")
    create_preview_grid(image_paths=image_paths, output_path=output_dir / "preview_grid.png")

    return labels
