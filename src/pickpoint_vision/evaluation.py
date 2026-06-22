"""Evaluation utilities for pick-point estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math
import statistics
import time

from pickpoint_vision.pose_estimation import (
    PoseEstimationResult,
    axis_orientation_error_deg,
    estimate_pose_from_mask_file,
)


@dataclass(frozen=True)
class MetricStats:
    """Summary statistics for one metric."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    p90: float

    def to_dict(self) -> dict[str, float]:
        """Convert metric stats to a serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class PerSampleEvaluation:
    """Evaluation values for one sample."""

    image_name: str
    shape: str
    gt_center_x: float
    gt_center_y: float
    pred_center_x: float
    pred_center_y: float
    gt_angle_deg: float
    pred_angle_deg_pca: float
    pred_angle_deg_min_area_rect: float
    center_error_px: float
    pca_orientation_error_deg: float
    min_area_rect_orientation_error_deg: float
    inference_time_ms: float

    def to_dict(self) -> dict[str, object]:
        """Convert sample evaluation to a serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    """Full evaluation report."""

    dataset_dir: str
    output_dir: str
    num_samples: int
    center_error_px: MetricStats
    pca_orientation_error_deg: MetricStats
    min_area_rect_orientation_error_deg: MetricStats
    inference_time_ms: MetricStats
    center_error_pass_threshold_px: float
    orientation_error_pass_threshold_deg: float
    center_pass_rate: float
    pca_orientation_pass_rate: float
    worst_center_error_sample: str
    worst_pca_orientation_error_sample: str
    per_sample_metrics_csv: str
    summary_json: str
    markdown_report: str

    def to_dict(self) -> dict[str, object]:
        """Convert report to a serializable dictionary."""
        data = asdict(self)
        data["center_error_px"] = self.center_error_px.to_dict()
        data["pca_orientation_error_deg"] = self.pca_orientation_error_deg.to_dict()
        data["min_area_rect_orientation_error_deg"] = (
            self.min_area_rect_orientation_error_deg.to_dict()
        )
        data["inference_time_ms"] = self.inference_time_ms.to_dict()
        return data


def compute_center_error_px(
    predicted_center: tuple[float, float],
    ground_truth_center: tuple[float, float],
) -> float:
    """Compute Euclidean center-point error in pixels."""
    pred_x, pred_y = predicted_center
    gt_x, gt_y = ground_truth_center
    return float(math.hypot(pred_x - gt_x, pred_y - gt_y))


def compute_pass_rate(values: list[float], threshold: float) -> float:
    """Return percentage of values less than or equal to a threshold."""
    if not values:
        raise ValueError("Cannot compute pass rate for an empty list.")

    passed = sum(value <= threshold for value in values)
    return round((passed / len(values)) * 100.0, 3)


def percentile(values: list[float], percentile_value: float) -> float:
    """Compute a simple nearest-rank percentile."""
    if not values:
        raise ValueError("Cannot compute percentile for an empty list.")

    if percentile_value < 0 or percentile_value > 100:
        raise ValueError("percentile_value must be between 0 and 100.")

    sorted_values = sorted(values)
    index = math.ceil((percentile_value / 100.0) * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return float(sorted_values[index])


def summarize_metric(values: list[float]) -> MetricStats:
    """Create summary statistics for one metric."""
    if not values:
        raise ValueError("Cannot summarize an empty metric list.")

    return MetricStats(
        mean=round(statistics.mean(values), 3),
        median=round(statistics.median(values), 3),
        std=round(statistics.pstdev(values), 3),
        min=round(min(values), 3),
        max=round(max(values), 3),
        p90=round(percentile(values, 90.0), 3),
    )


def evaluate_sample(
    label_row: dict[str, str],
    pose_result: PoseEstimationResult,
    inference_time_ms: float,
) -> PerSampleEvaluation:
    """Evaluate one pose result against one synthetic ground-truth label."""
    gt_center_x = float(label_row["center_x"])
    gt_center_y = float(label_row["center_y"])
    gt_angle_deg = float(label_row["angle_deg"])

    center_error = compute_center_error_px(
        predicted_center=(pose_result.center_x, pose_result.center_y),
        ground_truth_center=(gt_center_x, gt_center_y),
    )
    pca_orientation_error = axis_orientation_error_deg(
        pose_result.angle_deg_pca,
        gt_angle_deg,
    )
    min_area_rect_orientation_error = axis_orientation_error_deg(
        pose_result.angle_deg_min_area_rect,
        gt_angle_deg,
    )

    return PerSampleEvaluation(
        image_name=label_row["image_name"],
        shape=label_row["shape"],
        gt_center_x=round(gt_center_x, 3),
        gt_center_y=round(gt_center_y, 3),
        pred_center_x=pose_result.center_x,
        pred_center_y=pose_result.center_y,
        gt_angle_deg=round(gt_angle_deg, 3),
        pred_angle_deg_pca=pose_result.angle_deg_pca,
        pred_angle_deg_min_area_rect=pose_result.angle_deg_min_area_rect,
        center_error_px=round(center_error, 3),
        pca_orientation_error_deg=round(pca_orientation_error, 3),
        min_area_rect_orientation_error_deg=round(min_area_rect_orientation_error, 3),
        inference_time_ms=round(inference_time_ms, 3),
    )


def load_label_rows(labels_csv: str | Path) -> list[dict[str, str]]:
    """Load label rows from the synthetic labels CSV."""
    labels_csv = Path(labels_csv)
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Missing labels file: {labels_csv}. "
            "Run: py scripts\\create_synthetic_dataset.py --num-images 20 --clear"
        )

    with labels_csv.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_per_sample_csv(samples: list[PerSampleEvaluation], output_path: str | Path) -> Path:
    """Save per-sample evaluation metrics to CSV."""
    if not samples:
        raise ValueError("Cannot save an empty sample list.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(samples[0].to_dict().keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.to_dict())

    return output_path


def save_summary_json(report: EvaluationReport, output_path: str | Path) -> Path:
    """Save evaluation summary to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, indent=2)

    return output_path


def save_markdown_report(report: EvaluationReport, output_path: str | Path) -> Path:
    """Save a recruiter-friendly Markdown evaluation report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# Evaluation Report

## Dataset

- Dataset directory: `{report.dataset_dir}`
- Samples evaluated: **{report.num_samples}**

## Summary table

| Metric | Mean | Median | P90 | Max |
|---|---:|---:|---:|---:|
| Center error (px) | {report.center_error_px.mean:.3f} | {report.center_error_px.median:.3f} | {report.center_error_px.p90:.3f} | {report.center_error_px.max:.3f} |
| PCA orientation error (deg) | {report.pca_orientation_error_deg.mean:.3f} | {report.pca_orientation_error_deg.median:.3f} | {report.pca_orientation_error_deg.p90:.3f} | {report.pca_orientation_error_deg.max:.3f} |
| minAreaRect orientation error (deg) | {report.min_area_rect_orientation_error_deg.mean:.3f} | {report.min_area_rect_orientation_error_deg.median:.3f} | {report.min_area_rect_orientation_error_deg.p90:.3f} | {report.min_area_rect_orientation_error_deg.max:.3f} |
| Inference time (ms/image) | {report.inference_time_ms.mean:.3f} | {report.inference_time_ms.median:.3f} | {report.inference_time_ms.p90:.3f} | {report.inference_time_ms.max:.3f} |

## Pass-rate checks

| Check | Threshold | Pass rate |
|---|---:|---:|
| Center error | <= {report.center_error_pass_threshold_px:.3f} px | {report.center_pass_rate:.3f}% |
| PCA orientation error | <= {report.orientation_error_pass_threshold_deg:.3f} deg | {report.pca_orientation_pass_rate:.3f}% |

## Worst cases

- Worst center-error sample: `{report.worst_center_error_sample}`
- Worst PCA-orientation-error sample: `{report.worst_pca_orientation_error_sample}`

## Generated files

- Per-sample CSV: `{report.per_sample_metrics_csv}`
- Summary JSON: `{report.summary_json}`
- Markdown report: `{report.markdown_report}`

## Notes

This report evaluates the current synthetic-mask baseline. The values are expected to be strong because the segmentation masks are clean and the ground truth is generated programmatically. Later robustness tests will intentionally degrade the images to expose failure cases.
"""

    output_path.write_text(content, encoding="utf-8")
    return output_path


def create_evaluation_report(
    dataset_dir: str | Path,
    output_dir: str | Path,
    samples: list[PerSampleEvaluation],
    center_error_pass_threshold_px: float = 2.0,
    orientation_error_pass_threshold_deg: float = 5.0,
) -> EvaluationReport:
    """Create an evaluation report object from per-sample metrics."""
    if not samples:
        raise ValueError("Cannot create an evaluation report from an empty sample list.")

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    center_errors = [sample.center_error_px for sample in samples]
    pca_errors = [sample.pca_orientation_error_deg for sample in samples]
    rect_errors = [sample.min_area_rect_orientation_error_deg for sample in samples]
    inference_times = [sample.inference_time_ms for sample in samples]

    worst_center_sample = max(samples, key=lambda sample: sample.center_error_px)
    worst_pca_sample = max(samples, key=lambda sample: sample.pca_orientation_error_deg)

    per_sample_csv = output_dir / "per_sample_metrics.csv"
    summary_json = output_dir / "evaluation_summary.json"
    markdown_report = output_dir / "evaluation_report.md"

    return EvaluationReport(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        num_samples=len(samples),
        center_error_px=summarize_metric(center_errors),
        pca_orientation_error_deg=summarize_metric(pca_errors),
        min_area_rect_orientation_error_deg=summarize_metric(rect_errors),
        inference_time_ms=summarize_metric(inference_times),
        center_error_pass_threshold_px=center_error_pass_threshold_px,
        orientation_error_pass_threshold_deg=orientation_error_pass_threshold_deg,
        center_pass_rate=compute_pass_rate(center_errors, center_error_pass_threshold_px),
        pca_orientation_pass_rate=compute_pass_rate(
            pca_errors,
            orientation_error_pass_threshold_deg,
        ),
        worst_center_error_sample=worst_center_sample.image_name,
        worst_pca_orientation_error_sample=worst_pca_sample.image_name,
        per_sample_metrics_csv=str(per_sample_csv),
        summary_json=str(summary_json),
        markdown_report=str(markdown_report),
    )


def run_synthetic_evaluation(
    dataset_dir: str | Path,
    output_dir: str | Path,
    center_error_pass_threshold_px: float = 2.0,
    orientation_error_pass_threshold_deg: float = 5.0,
) -> EvaluationReport:
    """Evaluate pose estimation on a synthetic dataset and save reports."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_label_rows(dataset_dir / "labels.csv")
    if not rows:
        raise ValueError("No labels found for evaluation.")

    masks_dir = dataset_dir / "masks"
    samples: list[PerSampleEvaluation] = []

    for row in rows:
        image_name = row["image_name"]
        mask_path = masks_dir / image_name.replace(".png", "_mask.png")

        start_time = time.perf_counter()
        pose_result = estimate_pose_from_mask_file(mask_path=mask_path, image_name=image_name)
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        samples.append(
            evaluate_sample(
                label_row=row,
                pose_result=pose_result,
                inference_time_ms=inference_time_ms,
            )
        )

    report = create_evaluation_report(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        samples=samples,
        center_error_pass_threshold_px=center_error_pass_threshold_px,
        orientation_error_pass_threshold_deg=orientation_error_pass_threshold_deg,
    )

    save_per_sample_csv(samples=samples, output_path=report.per_sample_metrics_csv)
    save_summary_json(report=report, output_path=report.summary_json)
    save_markdown_report(report=report, output_path=report.markdown_report)

    return report
