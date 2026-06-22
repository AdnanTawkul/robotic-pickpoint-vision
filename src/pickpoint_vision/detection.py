"""YOLO-based object detection utilities.

This module provides an optional YOLO detection path for real RGB images.
It supports configurable confidence, image size, IoU, augmentation, max detections,
and CUDA-aware device selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import math
from typing import Any

import cv2
import numpy as np

from pickpoint_vision.utils import list_image_files
from pickpoint_vision.visualization import draw_label_background


@dataclass(frozen=True)
class DetectionResult:
    """Single object-detection result."""

    image_name: str
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, object]:
        """Convert detection result to a serializable dictionary."""
        return asdict(self)


def resolve_yolo_device(device: str = "auto") -> str:
    """Resolve a YOLO inference device safely.

    Returns:
        - "0" when CUDA is available and GPU was requested/auto-selected
        - "cpu" when CUDA is unavailable
        - another explicit non-CUDA device string unchanged

    Ultralytics accepts "0" for the first CUDA GPU and "cpu" for CPU.

    Important:
        A PC can have an NVIDIA GPU while PyTorch is still CPU-only. In that
        case, torch.cuda.is_available() is False and YOLO must use CPU until
        CUDA-enabled PyTorch is installed.
    """
    normalized = device.strip().lower()

    try:
        import torch
    except ImportError:
        return "cpu"

    cuda_available = torch.cuda.is_available()
    cuda_requested = normalized in {"auto", "", "cuda", "cuda:0", "gpu", "0"}

    if normalized in {"auto", ""}:
        return "0" if cuda_available else "cpu"

    if cuda_requested:
        return "0" if cuda_available else "cpu"

    return device


@dataclass(frozen=True)
class YoloInferenceConfig:
    """YOLO inference configuration."""

    model_name_or_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    image_size: int = 640
    iou_threshold: float = 0.70
    augment: bool = False
    max_detections: int = 100
    device: str = "auto"

    def to_predict_kwargs(self) -> dict[str, object]:
        """Convert config to Ultralytics predict keyword arguments."""
        return {
            "conf": self.confidence_threshold,
            "imgsz": self.image_size,
            "iou": self.iou_threshold,
            "augment": self.augment,
            "max_det": self.max_detections,
            "device": resolve_yolo_device(self.device),
            "verbose": False,
        }


def load_yolo_model(model_name_or_path: str = "yolov8n.pt") -> Any:
    """Load a YOLO model from Ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed. Install it with:\n"
            "  py -m pip install ultralytics\n"
            "or reinstall project dependencies with:\n"
            "  py -m pip install -e ."
        ) from exc

    return YOLO(model_name_or_path)


def _tensor_like_to_numpy(values: Any) -> np.ndarray:
    """Convert torch/numpy/list-like values to a NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return values.numpy()
    return np.asarray(values)


def _extract_class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    """Extract a class name from YOLO's names structure."""
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def detection_from_xyxy(
    image_name: str,
    class_id: int,
    class_name: str,
    confidence: float,
    xyxy: tuple[float, float, float, float],
) -> DetectionResult:
    """Create a DetectionResult from xyxy bounding-box coordinates."""
    x1, y1, x2, y2 = xyxy
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return DetectionResult(
        image_name=image_name,
        class_id=int(class_id),
        class_name=str(class_name),
        confidence=round(float(confidence), 4),
        x1=round(float(x1), 3),
        y1=round(float(y1), 3),
        x2=round(float(x2), 3),
        y2=round(float(y2), 3),
        center_x=round(float(x1 + width / 2.0), 3),
        center_y=round(float(y1 + height / 2.0), 3),
        width=round(float(width), 3),
        height=round(float(height), 3),
    )


def filter_detections(
    detections: list[DetectionResult],
    allowed_class_names: set[str] | None = None,
    min_confidence: float = 0.0,
) -> list[DetectionResult]:
    """Filter detections by class name and confidence."""
    filtered: list[DetectionResult] = []
    normalized_allowed = None
    if allowed_class_names is not None:
        normalized_allowed = {name.lower() for name in allowed_class_names}

    for detection in detections:
        if detection.confidence < min_confidence:
            continue
        if normalized_allowed is not None and detection.class_name.lower() not in normalized_allowed:
            continue
        filtered.append(detection)
    return filtered


def run_yolo_on_image(
    image_path: str | Path,
    model: Any,
    confidence_threshold: float = 0.25,
    allowed_class_names: set[str] | None = None,
    image_size: int = 640,
    iou_threshold: float = 0.70,
    augment: bool = False,
    max_detections: int = 100,
    device: str = "auto",
) -> list[DetectionResult]:
    """Run YOLO on one image and return parsed detections."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    config = YoloInferenceConfig(
        confidence_threshold=confidence_threshold,
        image_size=image_size,
        iou_threshold=iou_threshold,
        augment=augment,
        max_detections=max_detections,
        device=device,
    )

    predictions = model.predict(source=str(image_path), **config.to_predict_kwargs())
    if not predictions:
        return []

    prediction = predictions[0]
    names = prediction.names
    boxes = prediction.boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections: list[DetectionResult] = []
    for box in boxes:
        xyxy_array = _tensor_like_to_numpy(box.xyxy[0]).astype(float)
        class_id = int(_tensor_like_to_numpy(box.cls[0]).item())
        confidence = float(_tensor_like_to_numpy(box.conf[0]).item())
        class_name = _extract_class_name(names, class_id)

        detections.append(
            detection_from_xyxy(
                image_name=image_path.name,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                xyxy=(
                    float(xyxy_array[0]),
                    float(xyxy_array[1]),
                    float(xyxy_array[2]),
                    float(xyxy_array[3]),
                ),
            )
        )

    detections = filter_detections(
        detections=detections,
        allowed_class_names=allowed_class_names,
        min_confidence=confidence_threshold,
    )
    return sorted(detections, key=lambda detection: detection.confidence, reverse=True)


def run_yolo_on_folder(
    input_dir: str | Path,
    model: Any,
    confidence_threshold: float = 0.25,
    allowed_class_names: set[str] | None = None,
    max_images: int | None = None,
    image_size: int = 640,
    iou_threshold: float = 0.70,
    augment: bool = False,
    max_detections: int = 100,
    device: str = "auto",
) -> dict[Path, list[DetectionResult]]:
    """Run YOLO on a folder of images."""
    image_paths = list_image_files(input_dir)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    results: dict[Path, list[DetectionResult]] = {}
    for image_path in image_paths:
        results[image_path] = run_yolo_on_image(
            image_path=image_path,
            model=model,
            confidence_threshold=confidence_threshold,
            allowed_class_names=allowed_class_names,
            image_size=image_size,
            iou_threshold=iou_threshold,
            augment=augment,
            max_detections=max_detections,
            device=device,
        )
    return results


def draw_detection_results(image: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
    """Draw bounding boxes, labels, and detection centers."""
    annotated = image.copy()
    if not detections:
        draw_label_background(annotated, "No YOLO detections", origin=(15, 30))
        return annotated

    for detection in detections:
        x1 = int(round(detection.x1))
        y1 = int(round(detection.y1))
        x2 = int(round(detection.x2))
        y2 = int(round(detection.y2))
        center = (int(round(detection.center_x)), int(round(detection.center_y)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), thickness=2)
        cv2.circle(annotated, center, radius=5, color=(0, 0, 255), thickness=-1)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_y = max(25, y1 - 8)
        draw_label_background(annotated, label, origin=(max(10, x1), text_y))
    return annotated


def annotate_detection_file(
    image_path: str | Path,
    detections: list[DetectionResult],
    output_path: str | Path,
) -> Path:
    """Draw detections on an image and save the annotation."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    annotated = draw_detection_results(image=image, detections=detections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    return output_path


def save_detection_results_csv(
    detections_by_image: dict[Path, list[DetectionResult]],
    output_path: str | Path,
) -> Path:
    """Save all detection results to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DetectionResult("", -1, "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).to_dict().keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for image_path, detections in detections_by_image.items():
            if not detections:
                writer.writerow({key: "" for key in fieldnames} | {"image_name": image_path.name})
                continue
            for detection in detections:
                writer.writerow(detection.to_dict())
    return output_path


def create_detection_preview_grid(
    annotated_image_paths: list[Path],
    output_path: str | Path,
    max_images: int = 12,
    cell_width: int = 320,
    cell_height: int = 240,
) -> Path:
    """Create a grid from YOLO annotated images."""
    selected_paths = annotated_image_paths[:max_images]
    if not selected_paths:
        raise ValueError("No annotated detection images provided.")

    images: list[np.ndarray] = []
    for image_path in selected_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        images.append(cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA))
    if not images:
        raise ValueError("Could not read any annotated detection images.")

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
