"""Tests for evaluation utilities."""

from pathlib import Path

from pickpoint_vision.evaluation import (
    compute_center_error_px,
    compute_pass_rate,
    run_synthetic_evaluation,
    summarize_metric,
)
from pickpoint_vision.synthetic_data import create_synthetic_dataset


def test_compute_center_error_px() -> None:
    """Center error should use Euclidean distance."""
    error = compute_center_error_px(
        predicted_center=(3.0, 4.0),
        ground_truth_center=(0.0, 0.0),
    )
    assert error == 5.0


def test_summarize_metric() -> None:
    """Metric summary should compute useful statistics."""
    stats = summarize_metric([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.mean == 3.0
    assert stats.median == 3.0
    assert stats.min == 1.0
    assert stats.max == 5.0
    assert stats.p90 == 5.0


def test_compute_pass_rate() -> None:
    """Pass rate should return a percentage."""
    assert compute_pass_rate([0.5, 1.0, 2.5, 3.0], threshold=2.0) == 50.0


def test_run_synthetic_evaluation(tmp_path: Path) -> None:
    """Synthetic evaluation should save CSV, JSON, and Markdown reports."""
    dataset_dir = tmp_path / "synthetic"
    output_dir = tmp_path / "evaluation"

    create_synthetic_dataset(
        output_dir=dataset_dir,
        num_images=5,
        image_width=320,
        image_height=240,
        seed=12,
    )

    report = run_synthetic_evaluation(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert report.num_samples == 5
    assert Path(report.per_sample_metrics_csv).exists()
    assert Path(report.summary_json).exists()
    assert Path(report.markdown_report).exists()
    assert report.center_error_px.mean < 2.0
    assert report.pca_orientation_error_deg.mean < 5.0
    assert report.center_pass_rate >= 80.0
