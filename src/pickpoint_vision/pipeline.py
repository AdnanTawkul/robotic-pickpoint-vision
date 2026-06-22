"""End-to-end demo pipeline utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import statistics
import time

from pickpoint_vision.pose_estimation import (
    PoseEstimationResult,
    axis_orientation_error_deg,
    estimate_pose_from_mask_file,
)
from pickpoint_vision.synthetic_data import create_synthetic_dataset
from pickpoint_vision.visualization import annotate_pose_from_files, create_annotation_grid


@dataclass(frozen=True)
class DemoSampleResult:
    """Result summary for one demo sample."""

    image_name: str
    shape: str
    ground_truth_center_x: float
    ground_truth_center_y: float
    estimated_center_x: float
    estimated_center_y: float
    ground_truth_angle_deg: float
    estimated_angle_deg_pca: float
    estimated_angle_deg_min_area_rect: float
    center_error_px: float
    pca_orientation_error_deg: float
    min_area_rect_orientation_error_deg: float
    annotated_image_path: str
    inference_time_ms: float

    def to_dict(self) -> dict[str, object]:
        """Convert sample result to a serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class DemoSummary:
    """End-to-end demo summary."""

    dataset_dir: str
    annotated_dir: str
    metrics_path: str
    preview_grid_path: str
    num_samples: int
    mean_center_error_px: float
    max_center_error_px: float
    mean_pca_orientation_error_deg: float
    mean_min_area_rect_orientation_error_deg: float
    mean_inference_time_ms: float
    samples: list[DemoSampleResult]

    def to_dict(self) -> dict[str, object]:
        """Convert summary to a serializable dictionary."""
        data = asdict(self)
        data["samples"] = [sample.to_dict() for sample in self.samples]
        return data


def load_synthetic_labels(labels_csv: str | Path) -> list[dict[str, str]]:
    """Load synthetic ground-truth labels from CSV."""
    labels_csv = Path(labels_csv)
    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_csv}")

    with labels_csv.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def synthetic_dataset_is_ready(dataset_dir: str | Path) -> bool:
    """Check whether the synthetic dataset has the required files."""
    dataset_dir = Path(dataset_dir)
    return (
        (dataset_dir / "labels.csv").exists()
        and (dataset_dir / "images").exists()
        and (dataset_dir / "masks").exists()
    )


def run_synthetic_demo(
    dataset_dir: str | Path,
    annotated_dir: str | Path,
    metrics_path: str | Path,
    num_images: int = 20,
    regenerate_dataset: bool = False,
    seed: int = 42,
) -> DemoSummary:
    """Run the synthetic end-to-end demo.

    The demo:
    1. creates or loads the synthetic dataset
    2. estimates pose from each mask
    3. saves annotated images
    4. creates a preview grid
    5. saves a JSON summary with simple accuracy and speed values
    """
    dataset_dir = Path(dataset_dir)
    annotated_dir = Path(annotated_dir)
    metrics_path = Path(metrics_path)

    if regenerate_dataset or not synthetic_dataset_is_ready(dataset_dir):
        create_synthetic_dataset(
            output_dir=dataset_dir,
            num_images=num_images,
            seed=seed,
            clear_existing=True,
        )

    annotated_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    labels = load_synthetic_labels(dataset_dir / "labels.csv")
    if not labels:
        raise ValueError("No synthetic labels found.")

    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    sample_results: list[DemoSampleResult] = []
    annotated_paths: list[Path] = []

    for row in labels:
        image_name = row["image_name"]
        image_path = images_dir / image_name
        mask_path = masks_dir / image_name.replace(".png", "_mask.png")
        annotated_path = annotated_dir / image_name.replace(".png", "_demo_annotated.png")

        start_time = time.perf_counter()
        pose_result: PoseEstimationResult = estimate_pose_from_mask_file(
            mask_path=mask_path,
            image_name=image_name,
        )
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        annotate_pose_from_files(
            image_path=image_path,
            mask_path=mask_path,
            result=pose_result,
            output_path=annotated_path,
            orientation_source="pca",
        )

        gt_center_x = float(row["center_x"])
        gt_center_y = float(row["center_y"])
        gt_angle_deg = float(row["angle_deg"])

        center_error_px = (
            (pose_result.center_x - gt_center_x) ** 2
            + (pose_result.center_y - gt_center_y) ** 2
        ) ** 0.5
        pca_orientation_error_deg = axis_orientation_error_deg(
            pose_result.angle_deg_pca,
            gt_angle_deg,
        )
        min_area_rect_orientation_error_deg = axis_orientation_error_deg(
            pose_result.angle_deg_min_area_rect,
            gt_angle_deg,
        )

        sample_results.append(
            DemoSampleResult(
                image_name=image_name,
                shape=row["shape"],
                ground_truth_center_x=round(gt_center_x, 3),
                ground_truth_center_y=round(gt_center_y, 3),
                estimated_center_x=pose_result.center_x,
                estimated_center_y=pose_result.center_y,
                ground_truth_angle_deg=round(gt_angle_deg, 3),
                estimated_angle_deg_pca=pose_result.angle_deg_pca,
                estimated_angle_deg_min_area_rect=pose_result.angle_deg_min_area_rect,
                center_error_px=round(center_error_px, 3),
                pca_orientation_error_deg=round(pca_orientation_error_deg, 3),
                min_area_rect_orientation_error_deg=round(min_area_rect_orientation_error_deg, 3),
                annotated_image_path=str(annotated_path),
                inference_time_ms=round(inference_time_ms, 3),
            )
        )
        annotated_paths.append(annotated_path)

    preview_grid_path = create_annotation_grid(
        image_paths=annotated_paths,
        output_path=annotated_dir / "demo_grid.png",
    )

    center_errors = [sample.center_error_px for sample in sample_results]
    pca_errors = [sample.pca_orientation_error_deg for sample in sample_results]
    rect_errors = [sample.min_area_rect_orientation_error_deg for sample in sample_results]
    inference_times = [sample.inference_time_ms for sample in sample_results]

    summary = DemoSummary(
        dataset_dir=str(dataset_dir),
        annotated_dir=str(annotated_dir),
        metrics_path=str(metrics_path),
        preview_grid_path=str(preview_grid_path),
        num_samples=len(sample_results),
        mean_center_error_px=round(statistics.mean(center_errors), 3),
        max_center_error_px=round(max(center_errors), 3),
        mean_pca_orientation_error_deg=round(statistics.mean(pca_errors), 3),
        mean_min_area_rect_orientation_error_deg=round(statistics.mean(rect_errors), 3),
        mean_inference_time_ms=round(statistics.mean(inference_times), 3),
        samples=sample_results,
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(summary.to_dict(), file, indent=2)

    return summary
