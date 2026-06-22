"""Tests for YOLO detection utilities that do not require a YOLO model."""

from pathlib import Path

import cv2
import numpy as np

from pickpoint_vision.detection import (
    DetectionResult,
    YoloInferenceConfig,
    annotate_detection_file,
    detection_from_xyxy,
    draw_detection_results,
    filter_detections,
    resolve_yolo_device,
    save_detection_results_csv,
)


def test_detection_from_xyxy() -> None:
    """Bounding-box parsing should compute center and size."""
    detection = detection_from_xyxy(
        image_name="image.png",
        class_id=39,
        class_name="bottle",
        confidence=0.91,
        xyxy=(10.0, 20.0, 110.0, 80.0),
    )

    assert detection.center_x == 60.0
    assert detection.center_y == 50.0
    assert detection.width == 100.0
    assert detection.height == 60.0
    assert detection.confidence == 0.91


def test_yolo_inference_config_kwargs() -> None:
    """YOLO inference config should map cleanly to predict kwargs."""
    config = YoloInferenceConfig(
        confidence_threshold=0.1,
        image_size=1280,
        iou_threshold=0.5,
        augment=True,
        max_detections=25,
        device="cpu",
    )

    kwargs = config.to_predict_kwargs()

    assert kwargs["conf"] == 0.1
    assert kwargs["imgsz"] == 1280
    assert kwargs["iou"] == 0.5
    assert kwargs["augment"] is True
    assert kwargs["max_det"] == 25
    assert kwargs["device"] == "cpu"
    assert kwargs["verbose"] is False


def test_resolve_yolo_device_explicit_values() -> None:
    """Device resolution should be safe whether CUDA is installed or not."""
    assert resolve_yolo_device("cpu") == "cpu"
    assert resolve_yolo_device("cuda:0") in {"cpu", "0"}
    assert resolve_yolo_device("0") in {"cpu", "0"}


def test_filter_detections() -> None:
    """Detection filtering should respect class and confidence."""
    detections = [
        DetectionResult("a.png", 1, "bottle", 0.90, 0, 0, 10, 10, 5, 5, 10, 10),
        DetectionResult("a.png", 2, "chair", 0.80, 0, 0, 10, 10, 5, 5, 10, 10),
        DetectionResult("a.png", 3, "cup", 0.10, 0, 0, 10, 10, 5, 5, 10, 10),
    ]

    filtered = filter_detections(
        detections=detections,
        allowed_class_names={"bottle", "cup"},
        min_confidence=0.25,
    )

    assert len(filtered) == 1
    assert filtered[0].class_name == "bottle"


def test_draw_detection_results_changes_image() -> None:
    """Drawing detections should preserve shape and add visible overlays."""
    image = np.full((120, 160, 3), 220, dtype=np.uint8)
    detections = [
        DetectionResult("a.png", 1, "bottle", 0.90, 30, 20, 100, 90, 65, 55, 70, 70),
    ]

    annotated = draw_detection_results(image=image, detections=detections)

    assert annotated.shape == image.shape
    assert annotated.dtype == image.dtype
    assert np.mean(cv2.absdiff(image, annotated)) > 0.0


def test_save_and_annotate_detection_outputs(tmp_path: Path) -> None:
    """Detection annotations and CSV files should be saved."""
    image_path = tmp_path / "input.png"
    output_path = tmp_path / "annotated.png"
    csv_path = tmp_path / "detections.csv"

    image = np.full((120, 160, 3), 220, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    detections = [
        DetectionResult("input.png", 1, "bottle", 0.90, 30, 20, 100, 90, 65, 55, 70, 70),
    ]

    annotate_detection_file(
        image_path=image_path,
        detections=detections,
        output_path=output_path,
    )
    save_detection_results_csv(
        detections_by_image={image_path: detections},
        output_path=csv_path,
    )

    assert output_path.exists()
    assert csv_path.exists()
    assert "bottle" in csv_path.read_text(encoding="utf-8")
