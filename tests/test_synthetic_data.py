"""Tests for synthetic dataset generation."""

from pathlib import Path

import cv2

from pickpoint_vision.synthetic_data import create_synthetic_dataset, normalize_angle_deg


def test_normalize_angle_deg() -> None:
    """Angles should be normalized to [-90, 90)."""
    assert normalize_angle_deg(0.0) == 0.0
    assert normalize_angle_deg(90.0) == -90.0
    assert normalize_angle_deg(180.0) == 0.0
    assert normalize_angle_deg(-100.0) == 80.0


def test_create_synthetic_dataset(tmp_path: Path) -> None:
    """Dataset generation should create images, masks, and labels."""
    labels = create_synthetic_dataset(
        output_dir=tmp_path,
        num_images=3,
        image_width=320,
        image_height=240,
        seed=7,
    )

    assert len(labels) == 3
    assert (tmp_path / "labels.csv").exists()
    assert (tmp_path / "labels.json").exists()
    assert (tmp_path / "preview_grid.png").exists()

    image_files = sorted((tmp_path / "images").glob("*.png"))
    mask_files = sorted((tmp_path / "masks").glob("*.png"))

    assert len(image_files) == 3
    assert len(mask_files) == 3

    first_image = cv2.imread(str(image_files[0]))
    first_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)

    assert first_image is not None
    assert first_image.shape == (240, 320, 3)
    assert first_mask is not None
    assert first_mask.shape == (240, 320)
    assert first_mask.max() == 255
