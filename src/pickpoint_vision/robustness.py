"""Robustness transformations for image-processing stress tests.

Step 7 creates controlled visual degradations:
- Gaussian blur
- Gaussian noise
- brightness shifts
- contrast shifts
- partial occlusion

These transformations are used to create qualitative robustness examples now.
In the next step, we will evaluate how these transformations affect image-based
pose estimation.
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
class RobustnessVariantMetadata:
    """Metadata for one robustness-transformed image."""

    source_image_name: str
    variant_image_name: str
    transform_name: str
    severity: str
    output_path: str
    parameters_json: str

    def to_dict(self) -> dict[str, object]:
        """Convert metadata to a serializable dictionary."""
        return asdict(self)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Apply Gaussian blur with an odd kernel size."""
    if kernel_size <= 1:
        return image.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def adjust_brightness(image: np.ndarray, beta: float = 40.0) -> np.ndarray:
    """Adjust image brightness with an additive offset."""
    adjusted = image.astype(np.float32) + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, alpha: float = 1.4) -> np.ndarray:
    """Adjust contrast around mid-gray."""
    adjusted = (image.astype(np.float32) - 127.5) * alpha + 127.5
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_occlusion(
    image: np.ndarray,
    occlusion_fraction: float = 0.25,
    rng: random.Random | None = None,
    color: tuple[int, int, int] = (35, 35, 35),
) -> np.ndarray:
    """Add a rectangular occlusion to a random region of the image."""
    if rng is None:
        rng = random.Random()

    if not 0.0 < occlusion_fraction < 1.0:
        raise ValueError("occlusion_fraction must be between 0 and 1.")

    output = image.copy()
    image_height, image_width = output.shape[:2]

    occlusion_area = image_width * image_height * occlusion_fraction
    aspect_ratio = rng.uniform(0.6, 1.8)

    occlusion_width = int(round(math.sqrt(occlusion_area * aspect_ratio)))
    occlusion_height = int(round(occlusion_area / max(1, occlusion_width)))

    occlusion_width = max(12, min(occlusion_width, image_width - 1))
    occlusion_height = max(12, min(occlusion_height, image_height - 1))

    x0 = rng.randint(0, image_width - occlusion_width)
    y0 = rng.randint(0, image_height - occlusion_height)

    cv2.rectangle(
        output,
        (x0, y0),
        (x0 + occlusion_width, y0 + occlusion_height),
        color,
        thickness=-1,
    )

    return output


def create_default_robustness_variants(
    image: np.ndarray,
    seed: int = 42,
) -> list[tuple[str, str, dict[str, object], np.ndarray]]:
    """Create the default set of robustness variants for one image."""
    rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)

    variants: list[tuple[str, str, dict[str, object], np.ndarray]] = [
        (
            "blur",
            "mild",
            {"kernel_size": 7},
            apply_gaussian_blur(image, kernel_size=7),
        ),
        (
            "blur",
            "strong",
            {"kernel_size": 17},
            apply_gaussian_blur(image, kernel_size=17),
        ),
        (
            "noise",
            "mild",
            {"sigma": 12.0},
            apply_gaussian_noise(image, sigma=12.0, rng=noise_rng),
        ),
        (
            "noise",
            "strong",
            {"sigma": 32.0},
            apply_gaussian_noise(image, sigma=32.0, rng=noise_rng),
        ),
        (
            "brightness",
            "dark",
            {"beta": -55.0},
            adjust_brightness(image, beta=-55.0),
        ),
        (
            "brightness",
            "bright",
            {"beta": 45.0},
            adjust_brightness(image, beta=45.0),
        ),
        (
            "contrast",
            "low",
            {"alpha": 0.55},
            adjust_contrast(image, alpha=0.55),
        ),
        (
            "contrast",
            "high",
            {"alpha": 1.65},
            adjust_contrast(image, alpha=1.65),
        ),
        (
            "occlusion",
            "partial",
            {"occlusion_fraction": 0.18},
            apply_occlusion(image, occlusion_fraction=0.18, rng=rng),
        ),
    ]

    return variants


def save_metadata_csv(metadata: list[RobustnessVariantMetadata], output_path: str | Path) -> Path:
    """Save robustness metadata to CSV."""
    if not metadata:
        raise ValueError("Cannot save empty robustness metadata.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(metadata[0].to_dict().keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in metadata:
            writer.writerow(item.to_dict())

    return output_path


def save_metadata_json(metadata: list[RobustnessVariantMetadata], output_path: str | Path) -> Path:
    """Save robustness metadata to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump([item.to_dict() for item in metadata], file, indent=2)

    return output_path


def create_robustness_preview_grid(
    image_paths: list[Path],
    output_path: str | Path,
    max_images: int = 12,
    cell_width: int = 320,
    cell_height: int = 240,
) -> Path:
    """Create a grid of robustness examples."""
    selected_paths = image_paths[:max_images]
    if not selected_paths:
        raise ValueError("No image paths provided.")

    images: list[np.ndarray] = []
    for image_path in selected_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        images.append(cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA))

    if not images:
        raise ValueError("Could not read any robustness images.")

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


def create_robustness_dataset(
    source_images_dir: str | Path,
    output_dir: str | Path,
    max_source_images: int = 5,
    seed: int = 42,
    clear_existing: bool = False,
) -> list[RobustnessVariantMetadata]:
    """Create robustness variants from source images.

    Output layout:

        output_dir/
        ├── images/
        ├── robustness_metadata.csv
        ├── robustness_metadata.json
        └── robustness_preview_grid.png
    """
    source_images_dir = Path(source_images_dir)
    output_dir = Path(output_dir)
    output_images_dir = output_dir / "images"

    if not source_images_dir.exists():
        raise FileNotFoundError(f"Missing source image directory: {source_images_dir}")

    if clear_existing and output_images_dir.exists():
        for image_path in output_images_dir.glob("*.png"):
            image_path.unlink()

    output_images_dir.mkdir(parents=True, exist_ok=True)

    source_image_paths = sorted(source_images_dir.glob("*.png"))[:max_source_images]
    if not source_image_paths:
        raise ValueError(f"No PNG images found in: {source_images_dir}")

    metadata: list[RobustnessVariantMetadata] = []
    output_image_paths: list[Path] = []

    for source_index, source_image_path in enumerate(source_image_paths):
        image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        variants = create_default_robustness_variants(
            image=image,
            seed=seed + source_index,
        )

        for transform_name, severity, parameters, variant_image in variants:
            variant_image_name = (
                f"{source_image_path.stem}_{transform_name}_{severity}.png"
            )
            output_path = output_images_dir / variant_image_name

            cv2.imwrite(str(output_path), variant_image)

            metadata.append(
                RobustnessVariantMetadata(
                    source_image_name=source_image_path.name,
                    variant_image_name=variant_image_name,
                    transform_name=transform_name,
                    severity=severity,
                    output_path=str(output_path),
                    parameters_json=json.dumps(parameters, sort_keys=True),
                )
            )
            output_image_paths.append(output_path)

    save_metadata_csv(metadata, output_dir / "robustness_metadata.csv")
    save_metadata_json(metadata, output_dir / "robustness_metadata.json")
    create_robustness_preview_grid(
        image_paths=output_image_paths,
        output_path=output_dir / "robustness_preview_grid.png",
    )

    return metadata
