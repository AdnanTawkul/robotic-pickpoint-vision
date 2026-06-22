"""Integrated pick-point pipeline.

This module combines optional YOLO detection, real-image segmentation, contour pose
estimation, and annotated pick-point visualization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math
import time
from typing import Any

import cv2
import numpy as np

from pickpoint_vision.detection import DetectionResult, draw_detection_results, load_yolo_model, run_yolo_on_image
from pickpoint_vision.pose_estimation import PoseEstimationResult, estimate_pose_from_mask
from pickpoint_vision.real_segmentation import segment_real_object
from pickpoint_vision.utils import list_image_files
from pickpoint_vision.visualization import annotate_pose_result, create_annotation_grid, draw_label_background


@dataclass(frozen=True)
class IntegratedPickPointResult:
    image_name: str
    method: str
    detection_class_name: str
    detection_confidence: float | None
    detection_x1: float | None
    detection_y1: float | None
    detection_x2: float | None
    detection_y2: float | None
    center_x: float
    center_y: float
    pick_x: float
    pick_y: float
    angle_deg_pca: float
    angle_deg_min_area_rect: float
    contour_area: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    inference_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IntegratedImageResult:
    image_name: str
    success: bool
    failure_reason: str
    results: list[IntegratedPickPointResult]
    annotated_image_path: str
    inference_time_ms: float

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["results"] = [result.to_dict() for result in self.results]
        return data


def _clip_box(detection: DetectionResult, image_width: int, image_height: int, padding: int = 12) -> tuple[int, int, int, int]:
    x1 = max(0, int(math.floor(detection.x1)) - padding)
    y1 = max(0, int(math.floor(detection.y1)) - padding)
    x2 = min(image_width, int(math.ceil(detection.x2)) + padding)
    y2 = min(image_height, int(math.ceil(detection.y2)) + padding)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid clipped detection box.")
    return x1, y1, x2, y2


def segment_foreground_auto(image: np.ndarray) -> np.ndarray:
    return segment_real_object(image)


def _offset_pose_result(pose_result: PoseEstimationResult, image_name: str, x_offset: int, y_offset: int) -> PoseEstimationResult:
    return PoseEstimationResult(
        image_name=image_name,
        center_x=round(pose_result.center_x + x_offset, 3),
        center_y=round(pose_result.center_y + y_offset, 3),
        pick_x=round(pose_result.pick_x + x_offset, 3),
        pick_y=round(pose_result.pick_y + y_offset, 3),
        angle_deg_pca=pose_result.angle_deg_pca,
        angle_deg_min_area_rect=pose_result.angle_deg_min_area_rect,
        contour_area=pose_result.contour_area,
        bbox_x=int(pose_result.bbox_x + x_offset),
        bbox_y=int(pose_result.bbox_y + y_offset),
        bbox_width=pose_result.bbox_width,
        bbox_height=pose_result.bbox_height,
    )


def _full_image_mask_from_roi(roi_mask: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int], x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    image_height, image_width = image_shape[:2]
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = roi_mask
    return full_mask


def _result_from_pose(image_name: str, method: str, pose_result: PoseEstimationResult, inference_time_ms: float, detection: DetectionResult | None = None) -> IntegratedPickPointResult:
    return IntegratedPickPointResult(
        image_name=image_name,
        method=method,
        detection_class_name=detection.class_name if detection else "",
        detection_confidence=detection.confidence if detection else None,
        detection_x1=detection.x1 if detection else None,
        detection_y1=detection.y1 if detection else None,
        detection_x2=detection.x2 if detection else None,
        detection_y2=detection.y2 if detection else None,
        center_x=pose_result.center_x,
        center_y=pose_result.center_y,
        pick_x=pose_result.pick_x,
        pick_y=pose_result.pick_y,
        angle_deg_pca=pose_result.angle_deg_pca,
        angle_deg_min_area_rect=pose_result.angle_deg_min_area_rect,
        contour_area=pose_result.contour_area,
        bbox_x=pose_result.bbox_x,
        bbox_y=pose_result.bbox_y,
        bbox_width=pose_result.bbox_width,
        bbox_height=pose_result.bbox_height,
        inference_time_ms=round(inference_time_ms, 3),
    )


def _draw_integrated_result(image: np.ndarray, detections: list[DetectionResult], pose_masks_and_results: list[tuple[np.ndarray, PoseEstimationResult]], message: str = "") -> np.ndarray:
    annotated = draw_detection_results(image=image, detections=detections) if detections else image.copy()
    for mask, pose_result in pose_masks_and_results:
        annotated = annotate_pose_result(image=annotated, mask=mask, result=pose_result, orientation_source="pca")
    if message:
        draw_label_background(annotated, message, origin=(15, 30))
    return annotated


def run_integrated_pickpoint_on_image(
    image_path: str | Path,
    output_path: str | Path,
    model: Any | None = None,
    confidence_threshold: float = 0.25,
    allowed_class_names: set[str] | None = None,
    use_yolo: bool = True,
    fallback_to_opencv: bool = True,
    max_detections: int = 3,
    image_size: int = 640,
    iou_threshold: float = 0.70,
    augment: bool = False,
    device: str = "auto",
) -> IntegratedImageResult:
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return IntegratedImageResult(image_path.name, False, f"Could not read image: {image_path}", [], str(output_path), 0.0)

    image_height, image_width = image.shape[:2]
    start_time = time.perf_counter()

    detections: list[DetectionResult] = []
    if use_yolo and model is not None:
        detections = run_yolo_on_image(
            image_path=image_path,
            model=model,
            confidence_threshold=confidence_threshold,
            allowed_class_names=allowed_class_names,
            image_size=image_size,
            iou_threshold=iou_threshold,
            augment=augment,
            max_detections=max_detections,
            device=device,
        )[:max_detections]

    integrated_results: list[IntegratedPickPointResult] = []
    pose_masks_and_results: list[tuple[np.ndarray, PoseEstimationResult]] = []

    for detection in detections:
        try:
            x1, y1, x2, y2 = _clip_box(detection, image_width, image_height)
            roi = image[y1:y2, x1:x2]
            roi_mask = segment_foreground_auto(roi)
            roi_pose = estimate_pose_from_mask(roi_mask, image_name=image_path.name)
            full_pose = _offset_pose_result(roi_pose, image_path.name, x1, y1)
            full_mask = _full_image_mask_from_roi(roi_mask, image.shape, x1, y1, x2, y2)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            integrated_results.append(_result_from_pose(image_path.name, "yolo+opencv", full_pose, elapsed_ms, detection))
            pose_masks_and_results.append((full_mask, full_pose))
        except Exception:
            continue

    if not integrated_results and fallback_to_opencv:
        try:
            full_mask = segment_foreground_auto(image)
            pose_result = estimate_pose_from_mask(full_mask, image_name=image_path.name)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            integrated_results.append(_result_from_pose(image_path.name, "opencv_fallback", pose_result, elapsed_ms, None))
            pose_masks_and_results.append((full_mask, pose_result))
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            annotated = image.copy()
            draw_label_background(annotated, f"FAILED: {exc}", origin=(15, 30))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)
            return IntegratedImageResult(image_path.name, False, str(exc), [], str(output_path), round(elapsed_ms, 3))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    message = f"Integrated targets: {len(integrated_results)}" if integrated_results else "No integrated pick target"
    annotated = _draw_integrated_result(image, detections, pose_masks_and_results, message)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    return IntegratedImageResult(image_path.name, bool(integrated_results), "" if integrated_results else "No pick target estimated.", integrated_results, str(output_path), round(elapsed_ms, 3))


def save_integrated_results_csv(image_results: list[IntegratedImageResult], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(IntegratedPickPointResult("", "", "", None, None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0).to_dict().keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for image_result in image_results:
            if not image_result.results:
                writer.writerow({key: "" for key in fieldnames} | {"image_name": image_result.image_name, "inference_time_ms": image_result.inference_time_ms})
                continue
            for result in image_result.results:
                row = result.to_dict()
                for key, value in row.items():
                    if value is None:
                        row[key] = ""
                writer.writerow(row)
    return output_path


def save_integrated_summary_json(image_results: list[IntegratedImageResult], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump([image_result.to_dict() for image_result in image_results], file, indent=2)
    return output_path


def run_integrated_pickpoint_on_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    metrics_csv: str | Path,
    summary_json: str | Path,
    model_name_or_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
    allowed_class_names: set[str] | None = None,
    use_yolo: bool = True,
    fallback_to_opencv: bool = True,
    max_images: int | None = 10,
    max_detections: int = 3,
    image_size: int = 640,
    iou_threshold: float = 0.70,
    augment: bool = False,
    device: str = "auto",
) -> tuple[list[IntegratedImageResult], Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    image_paths = list_image_files(input_dir)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f"No images found in: {input_dir}")

    model = load_yolo_model(model_name_or_path) if use_yolo else None
    output_dir.mkdir(parents=True, exist_ok=True)

    image_results: list[IntegratedImageResult] = []
    annotated_paths: list[Path] = []
    for image_path in image_paths:
        output_path = output_dir / image_path.name.replace(image_path.suffix, "_integrated_annotated.png")
        image_result = run_integrated_pickpoint_on_image(
            image_path=image_path,
            output_path=output_path,
            model=model,
            confidence_threshold=confidence_threshold,
            allowed_class_names=allowed_class_names,
            use_yolo=use_yolo,
            fallback_to_opencv=fallback_to_opencv,
            max_detections=max_detections,
            image_size=image_size,
            iou_threshold=iou_threshold,
            augment=augment,
            device=device,
        )
        image_results.append(image_result)
        annotated_paths.append(output_path)

    save_integrated_results_csv(image_results, metrics_csv)
    save_integrated_summary_json(image_results, summary_json)
    grid_path = create_annotation_grid(annotated_paths, output_dir / "integrated_pickpoint_grid.png")
    return image_results, grid_path
