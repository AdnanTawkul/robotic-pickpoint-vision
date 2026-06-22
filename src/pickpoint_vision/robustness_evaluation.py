"""Quantitative robustness evaluation for image-based pose estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math
import statistics
import time

import cv2
import numpy as np

from pickpoint_vision.evaluation import compute_center_error_px, summarize_metric
from pickpoint_vision.pose_estimation import axis_orientation_error_deg, estimate_pose_from_mask
from pickpoint_vision.robustness import (
    RobustnessVariantMetadata,
    create_robustness_dataset,
)
from pickpoint_vision.segmentation import foreground_coverage, segment_dark_foreground
from pickpoint_vision.visualization import annotate_pose_result, create_annotation_grid, draw_label_background


@dataclass(frozen=True)
class RobustnessEvaluationSample:
    """Per-variant robustness evaluation result."""

    source_image_name: str
    variant_image_name: str
    transform_name: str
    severity: str
    success: bool
    failure_reason: str
    gt_center_x: float
    gt_center_y: float
    pred_center_x: float | None
    pred_center_y: float | None
    gt_angle_deg: float
    pred_angle_deg_pca: float | None
    center_error_px: float | None
    pca_orientation_error_deg: float | None
    foreground_coverage_percent: float | None
    inference_time_ms: float
    annotated_image_path: str

    def to_dict(self) -> dict[str, object]:
        """Convert sample to a serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RobustnessGroupSummary:
    """Summary for one transform/severity group."""

    transform_name: str
    severity: str
    num_samples: int
    successful_samples: int
    success_rate_percent: float
    mean_center_error_px: float | None
    max_center_error_px: float | None
    mean_pca_orientation_error_deg: float | None
    max_pca_orientation_error_deg: float | None
    mean_inference_time_ms: float

    def to_dict(self) -> dict[str, object]:
        """Convert group summary to a serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RobustnessEvaluationReport:
    """Full robustness evaluation report."""

    dataset_dir: str
    robustness_dir: str
    output_dir: str
    num_variants: int
    successful_variants: int
    success_rate_percent: float
    group_summaries: list[RobustnessGroupSummary]
    per_variant_metrics_csv: str
    summary_json: str
    markdown_report: str
    annotated_dir: str
    preview_grid_path: str

    def to_dict(self) -> dict[str, object]:
        """Convert report to a serializable dictionary."""
        data = asdict(self)
        data["group_summaries"] = [summary.to_dict() for summary in self.group_summaries]
        return data


def load_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Load CSV rows."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_robustness_metadata(metadata_csv: str | Path) -> list[RobustnessVariantMetadata]:
    """Load robustness metadata from CSV."""
    rows = load_csv_rows(metadata_csv)
    metadata: list[RobustnessVariantMetadata] = []

    for row in rows:
        metadata.append(
            RobustnessVariantMetadata(
                source_image_name=row["source_image_name"],
                variant_image_name=row["variant_image_name"],
                transform_name=row["transform_name"],
                severity=row["severity"],
                output_path=row["output_path"],
                parameters_json=row["parameters_json"],
            )
        )

    return metadata


def _safe_float(value: float | None) -> str | float:
    """Return an empty string for None so CSV files stay readable."""
    if value is None:
        return ""
    return value


def save_per_variant_metrics(
    samples: list[RobustnessEvaluationSample],
    output_path: str | Path,
) -> Path:
    """Save per-variant robustness metrics to CSV."""
    if not samples:
        raise ValueError("Cannot save empty robustness sample list.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(samples[0].to_dict().keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            row = sample.to_dict()
            for key, value in row.items():
                if value is None:
                    row[key] = _safe_float(value)
            writer.writerow(row)

    return output_path


def _save_failure_annotation(
    image: np.ndarray,
    output_path: str | Path,
    message: str,
) -> Path:
    """Save an image with a visible failure label."""
    annotated = image.copy()
    draw_label_background(annotated, message, origin=(15, 30))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    return output_path


def evaluate_robustness_variant(
    metadata: RobustnessVariantMetadata,
    label_row: dict[str, str],
    annotated_dir: str | Path,
) -> RobustnessEvaluationSample:
    """Evaluate one robustness variant using image-based segmentation."""
    image_path = Path(metadata.output_path)
    annotated_dir = Path(annotated_dir)
    annotated_path = annotated_dir / metadata.variant_image_name.replace(".png", "_annotated.png")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return RobustnessEvaluationSample(
            source_image_name=metadata.source_image_name,
            variant_image_name=metadata.variant_image_name,
            transform_name=metadata.transform_name,
            severity=metadata.severity,
            success=False,
            failure_reason=f"Could not read image: {image_path}",
            gt_center_x=float(label_row["center_x"]),
            gt_center_y=float(label_row["center_y"]),
            pred_center_x=None,
            pred_center_y=None,
            gt_angle_deg=float(label_row["angle_deg"]),
            pred_angle_deg_pca=None,
            center_error_px=None,
            pca_orientation_error_deg=None,
            foreground_coverage_percent=None,
            inference_time_ms=0.0,
            annotated_image_path=str(annotated_path),
        )

    gt_center_x = float(label_row["center_x"])
    gt_center_y = float(label_row["center_y"])
    gt_angle_deg = float(label_row["angle_deg"])

    start_time = time.perf_counter()
    try:
        mask = segment_dark_foreground(image)
        coverage = foreground_coverage(mask)
        pose_result = estimate_pose_from_mask(mask=mask, image_name=metadata.variant_image_name)
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        center_error = compute_center_error_px(
            predicted_center=(pose_result.center_x, pose_result.center_y),
            ground_truth_center=(gt_center_x, gt_center_y),
        )
        angle_error = axis_orientation_error_deg(pose_result.angle_deg_pca, gt_angle_deg)

        annotated = annotate_pose_result(
            image=image,
            mask=mask,
            result=pose_result,
            orientation_source="pca",
        )
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated)

        return RobustnessEvaluationSample(
            source_image_name=metadata.source_image_name,
            variant_image_name=metadata.variant_image_name,
            transform_name=metadata.transform_name,
            severity=metadata.severity,
            success=True,
            failure_reason="",
            gt_center_x=round(gt_center_x, 3),
            gt_center_y=round(gt_center_y, 3),
            pred_center_x=pose_result.center_x,
            pred_center_y=pose_result.center_y,
            gt_angle_deg=round(gt_angle_deg, 3),
            pred_angle_deg_pca=pose_result.angle_deg_pca,
            center_error_px=round(center_error, 3),
            pca_orientation_error_deg=round(angle_error, 3),
            foreground_coverage_percent=round(coverage, 3),
            inference_time_ms=round(inference_time_ms, 3),
            annotated_image_path=str(annotated_path),
        )

    except Exception as exc:
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        _save_failure_annotation(
            image=image,
            output_path=annotated_path,
            message=f"FAILED: {metadata.transform_name}/{metadata.severity}",
        )

        return RobustnessEvaluationSample(
            source_image_name=metadata.source_image_name,
            variant_image_name=metadata.variant_image_name,
            transform_name=metadata.transform_name,
            severity=metadata.severity,
            success=False,
            failure_reason=str(exc),
            gt_center_x=round(gt_center_x, 3),
            gt_center_y=round(gt_center_y, 3),
            pred_center_x=None,
            pred_center_y=None,
            gt_angle_deg=round(gt_angle_deg, 3),
            pred_angle_deg_pca=None,
            center_error_px=None,
            pca_orientation_error_deg=None,
            foreground_coverage_percent=None,
            inference_time_ms=round(inference_time_ms, 3),
            annotated_image_path=str(annotated_path),
        )


def _mean_or_none(values: list[float]) -> float | None:
    """Return rounded mean or None for an empty list."""
    if not values:
        return None
    return round(statistics.mean(values), 3)


def _max_or_none(values: list[float]) -> float | None:
    """Return rounded max or None for an empty list."""
    if not values:
        return None
    return round(max(values), 3)


def create_group_summaries(
    samples: list[RobustnessEvaluationSample],
) -> list[RobustnessGroupSummary]:
    """Group robustness metrics by transform and severity."""
    groups: dict[tuple[str, str], list[RobustnessEvaluationSample]] = {}

    for sample in samples:
        key = (sample.transform_name, sample.severity)
        groups.setdefault(key, []).append(sample)

    summaries: list[RobustnessGroupSummary] = []

    for (transform_name, severity), group_samples in sorted(groups.items()):
        successful = [sample for sample in group_samples if sample.success]
        center_errors = [
            float(sample.center_error_px)
            for sample in successful
            if sample.center_error_px is not None
        ]
        angle_errors = [
            float(sample.pca_orientation_error_deg)
            for sample in successful
            if sample.pca_orientation_error_deg is not None
        ]
        inference_times = [sample.inference_time_ms for sample in group_samples]

        summaries.append(
            RobustnessGroupSummary(
                transform_name=transform_name,
                severity=severity,
                num_samples=len(group_samples),
                successful_samples=len(successful),
                success_rate_percent=round((len(successful) / len(group_samples)) * 100.0, 3),
                mean_center_error_px=_mean_or_none(center_errors),
                max_center_error_px=_max_or_none(center_errors),
                mean_pca_orientation_error_deg=_mean_or_none(angle_errors),
                max_pca_orientation_error_deg=_max_or_none(angle_errors),
                mean_inference_time_ms=round(statistics.mean(inference_times), 3),
            )
        )

    return summaries


def save_robustness_summary_json(
    report: RobustnessEvaluationReport,
    output_path: str | Path,
) -> Path:
    """Save robustness summary JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, indent=2)

    return output_path


def save_robustness_markdown_report(
    report: RobustnessEvaluationReport,
    samples: list[RobustnessEvaluationSample],
    output_path: str | Path,
) -> Path:
    """Save a Markdown robustness report."""
    successful_samples = [sample for sample in samples if sample.success]

    if successful_samples:
        center_stats = summarize_metric(
            [float(sample.center_error_px) for sample in successful_samples if sample.center_error_px is not None]
        )
        angle_stats = summarize_metric(
            [
                float(sample.pca_orientation_error_deg)
                for sample in successful_samples
                if sample.pca_orientation_error_deg is not None
            ]
        )
        overall_summary = f"""## Overall successful-variant metrics

| Metric | Mean | Median | P90 | Max |
|---|---:|---:|---:|---:|
| Center error (px) | {center_stats.mean:.3f} | {center_stats.median:.3f} | {center_stats.p90:.3f} | {center_stats.max:.3f} |
| PCA orientation error (deg) | {angle_stats.mean:.3f} | {angle_stats.median:.3f} | {angle_stats.p90:.3f} | {angle_stats.max:.3f} |
"""
    else:
        overall_summary = "## Overall successful-variant metrics\n\nNo successful variants were available.\n"

    group_rows = []
    for group in report.group_summaries:
        group_rows.append(
            "| "
            f"{group.transform_name} | "
            f"{group.severity} | "
            f"{group.success_rate_percent:.1f}% | "
            f"{_format_optional_float(group.mean_center_error_px)} | "
            f"{_format_optional_float(group.max_center_error_px)} | "
            f"{_format_optional_float(group.mean_pca_orientation_error_deg)} | "
            f"{_format_optional_float(group.max_pca_orientation_error_deg)} |"
        )

    failed_samples = [sample for sample in samples if not sample.success]
    failed_lines = "\n".join(
        f"- `{sample.variant_image_name}`: {sample.failure_reason}"
        for sample in failed_samples[:10]
    )
    if not failed_lines:
        failed_lines = "- No hard failures."

    content = f"""# Robustness Evaluation Report

## Dataset

- Dataset directory: `{report.dataset_dir}`
- Robustness directory: `{report.robustness_dir}`
- Variants evaluated: **{report.num_variants}**
- Successful variants: **{report.successful_variants}**
- Success rate: **{report.success_rate_percent:.1f}%**

{overall_summary}
## Grouped robustness results

| Transform | Severity | Success rate | Mean center error | Max center error | Mean angle error | Max angle error |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(group_rows)}

## Hard failures

{failed_lines}

## Generated files

- Per-variant CSV: `{report.per_variant_metrics_csv}`
- Summary JSON: `{report.summary_json}`
- Markdown report: `{report.markdown_report}`
- Annotated outputs: `{report.annotated_dir}`
- Preview grid: `{report.preview_grid_path}`

## Interpretation

This evaluation uses classical image segmentation on degraded synthetic RGB images. The clean-mask baseline is expected to perform better. Large errors under occlusion or low contrast are useful failure cases, not necessarily project problems. They show where a learned detector/segmenter or better preprocessing would be needed in a real robotic perception pipeline.
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    return output_path


def _format_optional_float(value: float | None) -> str:
    """Format optional float for Markdown."""
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def run_robustness_evaluation(
    dataset_dir: str | Path,
    robustness_dir: str | Path,
    output_dir: str | Path,
    create_variants_if_missing: bool = True,
    max_source_images: int = 5,
    seed: int = 42,
) -> RobustnessEvaluationReport:
    """Run quantitative robustness evaluation."""
    dataset_dir = Path(dataset_dir)
    robustness_dir = Path(robustness_dir)
    output_dir = Path(output_dir)
    annotated_dir = output_dir / "annotated"

    labels_csv = dataset_dir / "labels.csv"
    source_images_dir = dataset_dir / "images"
    metadata_csv = robustness_dir / "robustness_metadata.csv"

    if create_variants_if_missing and not metadata_csv.exists():
        create_robustness_dataset(
            source_images_dir=source_images_dir,
            output_dir=robustness_dir,
            max_source_images=max_source_images,
            seed=seed,
            clear_existing=True,
        )

    labels = load_csv_rows(labels_csv)
    label_by_image_name = {row["image_name"]: row for row in labels}

    metadata_items = load_robustness_metadata(metadata_csv)
    if not metadata_items:
        raise ValueError("No robustness metadata found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    samples: list[RobustnessEvaluationSample] = []

    for metadata in metadata_items:
        if metadata.source_image_name not in label_by_image_name:
            raise KeyError(f"Missing label for source image: {metadata.source_image_name}")

        samples.append(
            evaluate_robustness_variant(
                metadata=metadata,
                label_row=label_by_image_name[metadata.source_image_name],
                annotated_dir=annotated_dir,
            )
        )

    per_variant_csv = output_dir / "per_variant_metrics.csv"
    summary_json = output_dir / "robustness_summary.json"
    markdown_report = output_dir / "robustness_report.md"

    save_per_variant_metrics(samples=samples, output_path=per_variant_csv)

    annotated_paths = [Path(sample.annotated_image_path) for sample in samples]
    preview_grid_path = create_annotation_grid(
        image_paths=annotated_paths,
        output_path=output_dir / "robustness_evaluation_grid.png",
    )

    successful_count = sum(sample.success for sample in samples)

    report = RobustnessEvaluationReport(
        dataset_dir=str(dataset_dir),
        robustness_dir=str(robustness_dir),
        output_dir=str(output_dir),
        num_variants=len(samples),
        successful_variants=successful_count,
        success_rate_percent=round((successful_count / len(samples)) * 100.0, 3),
        group_summaries=create_group_summaries(samples),
        per_variant_metrics_csv=str(per_variant_csv),
        summary_json=str(summary_json),
        markdown_report=str(markdown_report),
        annotated_dir=str(annotated_dir),
        preview_grid_path=str(preview_grid_path),
    )

    save_robustness_summary_json(report=report, output_path=summary_json)
    save_robustness_markdown_report(
        report=report,
        samples=samples,
        output_path=markdown_report,
    )

    return report
