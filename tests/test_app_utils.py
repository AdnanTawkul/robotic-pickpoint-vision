"""Tests for Streamlit app helper utilities."""

from pathlib import Path

import cv2
import numpy as np

from pickpoint_vision.app_utils import (
    parse_class_filter,
    read_image_as_rgb,
    sanitize_filename,
    save_uploaded_image_bytes,
)


def test_sanitize_filename() -> None:
    """Filename sanitization should remove unsafe characters."""
    assert sanitize_filename("hello world.jpg") == "hello_world.jpg"
    assert sanitize_filename("../bad/name.png") == "bad_name.png"
    assert sanitize_filename("   ") == "uploaded_image.png"


def test_parse_class_filter() -> None:
    """Class filter parsing should support comma-separated class names."""
    assert parse_class_filter("") is None
    assert parse_class_filter("bottle, cup, scissors") == {"bottle", "cup", "scissors"}
    assert parse_class_filter("bottle\ncup") == {"bottle", "cup"}


def test_save_uploaded_image_bytes_and_read_rgb(tmp_path: Path) -> None:
    """Uploaded image bytes should save and load as RGB."""
    image = np.full((40, 60, 3), 0, dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), image)

    upload = save_uploaded_image_bytes(
        image_bytes=source_path.read_bytes(),
        original_filename="test upload.png",
        output_dir=tmp_path / "uploads",
    )

    saved_path = Path(upload.saved_path)
    rgb = read_image_as_rgb(saved_path)

    assert saved_path.exists()
    assert upload.safe_filename == "test_upload.png"
    assert rgb.shape == (40, 60, 3)
    assert tuple(rgb[0, 0]) == (30, 20, 10)
