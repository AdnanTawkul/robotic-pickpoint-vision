"""Tests for robustness transformations."""

from pathlib import Path

import numpy as np

from pickpoint_vision.robustness import (
    adjust_brightness,
    adjust_contrast,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_occlusion,
    create_robustness_dataset,
)
from pickpoint_vision.synthetic_data import create_synthetic_dataset


def test_robustness_transforms_preserve_shape_and_dtype() -> None:
    """Robustness transformations should preserve image shape and dtype."""
    image = np.full((120, 160, 3), 128, dtype=np.uint8)

    variants = [
        apply_gaussian_blur(image, kernel_size=7),
        apply_gaussian_noise(image, sigma=10.0, rng=np.random.default_rng(1)),
        adjust_brightness(image, beta=30.0),
        adjust_contrast(image, alpha=1.4),
        apply_occlusion(image, occlusion_fraction=0.2),
    ]

    for variant in variants:
        assert variant.shape == image.shape
        assert variant.dtype == image.dtype


def test_create_robustness_dataset(tmp_path: Path) -> None:
    """Robustness dataset creation should save images and metadata."""
    synthetic_dir = tmp_path / "synthetic"
    robustness_dir = tmp_path / "robustness"

    create_synthetic_dataset(
        output_dir=synthetic_dir,
        num_images=2,
        image_width=320,
        image_height=240,
        seed=5,
    )

    metadata = create_robustness_dataset(
        source_images_dir=synthetic_dir / "images",
        output_dir=robustness_dir,
        max_source_images=2,
        seed=10,
    )

    assert len(metadata) == 18
    assert (robustness_dir / "robustness_metadata.csv").exists()
    assert (robustness_dir / "robustness_metadata.json").exists()
    assert (robustness_dir / "robustness_preview_grid.png").exists()
    assert len(list((robustness_dir / "images").glob("*.png"))) == 18
