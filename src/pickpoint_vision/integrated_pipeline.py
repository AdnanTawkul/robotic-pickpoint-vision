"""Integrated pick-point pipeline.

Step 10 combines:
- optional YOLO detection
- OpenCV foreground segmentation
- contour-based pose estimation
- annotated pick-point visualization

Step 13A adds stronger YOLO inference controls and passes them through the
integrated CLI and Streamlit app.
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

from pickpoint_vision.detection import (
    DetectionResult,
    draw_detection_results,
    load_yolo_model,
    run_yolo_on_image,
)
from pickpoint_vision.pose_estimation import PoseEstimationResult, estimate_pose_from_mask
from pickpoint_vision.utils import list_image_files
from pickpoint_vision.visualization import (
    annotate_pose_result,
    create_annotation_grid,
    draw_label_background,
)


@dataclass(frozen=True)
class IntegratedPickPointResult:
    """Integrated result for one estimated pick target."""

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
        """Convert result to a serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class IntegratedImageResult:
    """All integrated results for one image."""

    image_name: str
    success: bool
    failure_reason: str
    results: list[IntegratedPickPointResult]
    annotated_image_path: str
    inference_time_ms: float

    def to_dict(self) -> dict[str, object]:
        """Convert image result to a serializable dictionary."""
        data = asdict(self)
        data["results"] = [result.to_dict() for result in self.results]
        return data


def _clip_box(
    detection: DetectionResult,
    image_width: int,
    image_height: int,
    padding: int = 8,
) -> tuple[int, int, int, int]:
    """Clip a detection box to image boundaries and apply padding."""
    x1 = max(0, int(math.floor(detection.x1)) - padding)
    y1 = max(0, int(math.floor(detection.y1)) - padding)
    x2 = min(image_width, int(math.ceil(detection.x2)) + padding)
    y2 = min(image_height, int(math.ceil(detection.y2)) + padding)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid clipped detection box.")

    return x1, y1, x2, y2


def _mask_score(mask: np.ndarray) -> float:
    """Score a candidate binary mask."""
    height, width = mask.shape[:2]
    image_area = float(height * width)
    foreground = np.count_nonzero(mask)
    area_fraction = foreground / max(1.0, image_area)

    if area_fraction < 0.005 or area_fraction > 0.85:
        return -1.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1.0

    largest = max(contours, key=cv2.contourArea)
    largest_area = float(cv2.contourArea(largest))
    if largest_area <= 0.0:
        return -1.0

    dominance = largest_area / max(1.0, float(foreground))

    moments = cv2.moments(largest)
    if abs(moments["m00"]) < 1e-6:
        centrality = 0.0
    else:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        dx = abs(cx - width / 2.0) / max(1.0, width / 2.0)
        dy = abs(cy - height / 2.0) / max(1.0, height / 2.0)
        centrality = 1.0 - min(1.0, (dx + dy) / 2.0)

    area_balance = 1.0 - abs(area_fraction - 0.28)
    return float(dominance * 0.55 + centrality * 0.30 + area_balance * 0.15)


def segment_foreground_auto(image: np.ndarray) -> np.ndarray:
    """Segment foreground using automatic polarity selection."""
    if image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image.copy()

    if grayscale.dtype != np.uint8:
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)

    grayscale = cv2.GaussianBlur(grayscale, (5, 5), sigmaX=0)

    candidates: list[np.ndarray] = []

    for threshold_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
        _, mask = cv2.threshold(
            grayscale,
            0,
            255,
            threshold_type + cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        candidates.append(cleaned)

    scored = [(candidate, _mask_score(candidate)) for candidate in candidates]
    best_mask, best_score = max(scored, key=lambda item: item[1])

    if best_score < 0:
        raise ValueError("Could not create a usable foreground mask.")

    return best_mask


def _offset_pose_result(
    pose_result: PoseEstimationResult,
    image_name: str,
    x_offset: int,
    y_offset: int,
) -> PoseEstimationResult:
    """Convert ROI-local pose coordinates to full-image coordinates."""
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


def _full_image_mask_from_roi(
    roi_mask: np.ndarray,
    image_shape: tuple[int, int, int] | tuple[int, int],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Place an ROI mask into a full-image-sized mask."""
    image_height, image_width = image_shape[:2]
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = roi_mask
    return full_mask


def _result_from_pose(
    image_name: str,
    method: str,
    pose_result: PoseEstimationResult,
    inference_time_ms: float,
    detection: DetectionResult | None = None,
) -> IntegratedPickPointResult:
    """Create an integrated result from pose and optional detection."""
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


def _draw_integrated_result(
    image: np.ndarray,
    detections: list[DetectionResult],
    pose_masks_and_results: list[tuple[np.ndarray, PoseEstimationResult]],
    message: str = "",
) -> np.ndarray:
    """Draw detections and pose results on one image."""
    annotated = draw_detection_results(image=image, detections=detections) if detections else image.copy()

    for mask, pose_result in pose_masks_and_results:
        annotated = annotate_pose_result(
            image=annotated,
            mask=mask,
            result=pose_result,
            orientation_source="pca",
        )

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
) -> IntegratedImageResult:
    """Run the integrated pick-point pipeline on one image."""
    image_path = Path(image_path)
    output_path = Path(output_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return IntegratedImageResult(
            image_name=image_path.name,
            success=False,
            failure_reason=f"Could not read image: {image_path}",
            results=[],
            annotated_image_path=str(output_path),
            inference_time_ms=0.0,
        )

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
        )
        detections = sorted(detections, key=lambda item: item.confidence, reverse=True)[:max_detections]

    integrated_results: list[IntegratedPickPointResult] = []
    pose_masks_and_results: list[tuple[np.ndarray, PoseEstimationResult]] = []

    for detection in detections:
        try:
            x1, y1, x2, y2 = _clip_box(
                detection=detection,
                image_width=image_width,
                image_height=image_height,
            )
            roi = image[y1:y2, x1:x2]
            roi_mask = segment_foreground_auto(roi)
            roi_pose = estimate_pose_from_mask(roi_mask, image_name=image_path.name)
            full_pose = _offset_pose_result(
                pose_result=roi_pose,
                image_name=image_path.name,
                x_offset=x1,
                y_offset=y1,
            )
            full_mask = _full_image_mask_from_roi(
                roi_mask=roi_mask,
                image_shape=image.shape,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            integrated_results.append(
                _result_from_pose(
                    image_name=image_path.name,
                    method="yolo+opencv",
                    pose_result=full_pose,
                    detection=detection,
                    inference_time_ms=elapsed_ms,
                )
            )
            pose_masks_and_results.append((full_mask, full_pose))

        except Exception:
            continue

    if not integrated_results and fallback_to_opencv:
        try:
            full_mask = segment_foreground_auto(image)
            pose_result = estimate_pose_from_mask(full_mask, image_name=image_path.name)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            integrated_results.append(
                _result_from_pose(
                    image_name=image_path.name,
                    method="opencv_fallback",
                    pose_result=pose_result,
                    detection=None,
                    inference_time_ms=elapsed_ms,
                )
            )
            pose_masks_and_results.append((full_mask, pose_result))
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            annotated = image.copy()
            draw_label_background(annotated, f"FAILED: {exc}", origin=(15, 30))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)

            return IntegratedImageResult(
                image_name=image_path.name,
                success=False,
                failure_reason=str(exc),
                results=[],
                annotated_image_path=str(output_path),
                inference_time_ms=round(elapsed_ms, 3),
            )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    message = f"Integrated targets: {len(integrated_results)}" if integrated_results else "No integrated pick target"

    annotated = _draw_integrated_result(
        image=image,
        detections=detections,
        pose_masks_and_results=pose_masks_and_results,
        message=message,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    return IntegratedImageResult(
        image_name=image_path.name,
        success=bool(integrated_results),
        failure_reason="" if integrated_results else "No pick target estimated.",
        results=integrated_results,
        annotated_image_path=str(output_path),
        inference_time_ms=round(elapsed_ms, 3),
    )


def save_integrated_results_csv(
    image_results: list[IntegratedImageResult],
    output_path: str | Path,
) -> Path:
    """Save integrated pick-point results to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(
        IntegratedPickPointResult(
            image_name="",
            method="",
            detection_class_name="",
            detection_confidence=None,
            detection_x1=None,
            detection_y1=None,
            detection_x2=None,
            detection_y2=None,
            center_x=0.0,
            center_y=0.0,
            pick_x=0.0,
            pick_y=0.0,
            angle_deg_pca=0.0,
            angle_deg_min_area_rect=0.0,
            contour_area=0.0,
            bbox_x=0,
            bbox_y=0,
            bbox_width=0,
            bbox_height=0,
            inference_time_ms=0.0,
        ).to_dict().keys()
    )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for image_result in image_results:
            if not image_result.results:
                writer.writerow(
                    {
                        "image_name": image_result.image_name,
                        "method": "",
                        "detection_class_name": "",
                        "detection_confidence": "",
                        "detection_x1": "",
                        "detection_y1": "",
                        "detection_x2": "",
                        "detection_y2": "",
                        "center_x": "",
                        "center_y": "",
                        "pick_x": "",
                        "pick_y": "",
                        "angle_deg_pca": "",
                        "angle_deg_min_area_rect": "",
                        "contour_area": "",
                        "bbox_x": "",
                        "bbox_y": "",
                        "bbox_width": "",
                        "bbox_height": "",
                        "inference_time_ms": image_result.inference_time_ms,
                    }
                )
                continue

            for result in image_result.results:
                row = result.to_dict()
                for key, value in row.items():
                    if value is None:
                        row[key] = ""
                writer.writerow(row)

    return output_path


def save_integrated_summary_json(
    image_results: list[IntegratedImageResult],
    output_path: str | Path,
) -> Path:
    """Save integrated pipeline results to JSON."""
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
) -> tuple[list[IntegratedImageResult], Path]:
    """Run the integrated pipeline on a folder of images."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    image_paths = list_image_files(input_dir)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    if not image_paths:
        raise ValueError(f"No images found in: {input_dir}")

    model = None
    if use_yolo:
        model = load_yolo_model(model_name_or_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_results: list[IntegratedImageResult] = []
    annotated_paths: list[Path] = []

    for image_path in image_paths:
        output_path = output_dir / image_path.name.replace(
            image_path.suffix,
            "_integrated_annotated.png",
        )
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
        )
        image_results.append(image_result)
        annotated_paths.append(output_path)

    save_integrated_results_csv(image_results=image_results, output_path=metrics_csv)
    save_integrated_summary_json(image_results=image_results, output_path=summary_json)

    grid_path = create_annotation_grid(
        image_paths=annotated_paths,
        output_path=output_dir / "integrated_pickpoint_grid.png",
    )

    return image_results, grid_path
