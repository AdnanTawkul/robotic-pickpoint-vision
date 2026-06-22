"""Tests for the integrated pick-point pipeline."""

from pathlib import Path

import cv2
import numpy as np

from pickpoint_vision.integrated_pipeline import (
    run_integrated_pickpoint_on_folder,
    run_integrated_pickpoint_on_image,
    segment_foreground_auto,
)
from pickpoint_vision.pose_estimation import estimate_pose_from_mask
from pickpoint_vision.synthetic_data import create_synthetic_dataset


def test_segment_foreground_auto_bright_object_on_dark_background() -> None:
    """Auto segmentation should handle bright objects on dark backgrounds."""
    image = np.full((240, 320, 3), 30, dtype=np.uint8)
    cv2.ellipse(image, (160, 120), (70, 25), 30, 0, 360, (230, 230, 230), thickness=-1)

    mask = segment_foreground_auto(image)
    result = estimate_pose_from_mask(mask)

    assert abs(result.center_x - 160.0) < 5.0
    assert abs(result.center_y - 120.0) < 5.0


def test_run_integrated_pickpoint_on_image_opencv_fallback(tmp_path: Path) -> None:
    """OpenCV fallback should estimate a pick point without YOLO."""
    image_path = tmp_path / "object.png"
    output_path = tmp_path / "object_integrated.png"

    image = np.full((240, 320, 3), 230, dtype=np.uint8)
    rect = ((160.0, 120.0), (130.0, 50.0), 25.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(image, [box], contourIdx=-1, color=(60, 120, 160), thickness=-1)
    cv2.imwrite(str(image_path), image)

    result = run_integrated_pickpoint_on_image(
        image_path=image_path,
        output_path=output_path,
        model=None,
        use_yolo=False,
        fallback_to_opencv=True,
    )

    assert result.success
    assert output_path.exists()
    assert len(result.results) == 1
    assert result.results[0].method == "opencv_fallback"
    assert abs(result.results[0].center_x - 160.0) < 5.0


def test_run_integrated_pickpoint_on_folder(tmp_path: Path) -> None:
    """Folder-level integrated pipeline should save reports and a grid."""
    dataset_dir = tmp_path / "synthetic"
    output_dir = tmp_path / "annotated"
    metrics_csv = tmp_path / "metrics" / "integrated.csv"
    summary_json = tmp_path / "metrics" / "integrated.json"

    create_synthetic_dataset(
        output_dir=dataset_dir,
        num_images=3,
        image_width=320,
        image_height=240,
        seed=44,
    )

    image_results, grid_path = run_integrated_pickpoint_on_folder(
        input_dir=dataset_dir / "images",
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        summary_json=summary_json,
        use_yolo=False,
        fallback_to_opencv=True,
        max_images=3,
    )

    assert len(image_results) == 3
    assert all(result.success for result in image_results)
    assert metrics_csv.exists()
    assert summary_json.exists()
    assert Path(grid_path).exists()
    assert len(list(output_dir.glob("*_integrated_annotated.png"))) == 3
