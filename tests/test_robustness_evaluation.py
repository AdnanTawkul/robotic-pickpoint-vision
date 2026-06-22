"""Tests for quantitative robustness evaluation."""

from pathlib import Path

from pickpoint_vision.robustness import create_robustness_dataset
from pickpoint_vision.robustness_evaluation import run_robustness_evaluation
from pickpoint_vision.synthetic_data import create_synthetic_dataset


def test_run_robustness_evaluation(tmp_path: Path) -> None:
    """Robustness evaluation should create reports and annotations."""
    dataset_dir = tmp_path / "synthetic"
    robustness_dir = tmp_path / "robustness_variants"
    output_dir = tmp_path / "robustness_eval"

    create_synthetic_dataset(
        output_dir=dataset_dir,
        num_images=2,
        image_width=320,
        image_height=240,
        seed=5,
    )
    create_robustness_dataset(
        source_images_dir=dataset_dir / "images",
        output_dir=robustness_dir,
        max_source_images=2,
        seed=7,
    )

    report = run_robustness_evaluation(
        dataset_dir=dataset_dir,
        robustness_dir=robustness_dir,
        output_dir=output_dir,
        create_variants_if_missing=False,
    )

    assert report.num_variants == 18
    assert report.successful_variants > 0
    assert Path(report.per_variant_metrics_csv).exists()
    assert Path(report.summary_json).exists()
    assert Path(report.markdown_report).exists()
    assert Path(report.preview_grid_path).exists()
    assert len(report.group_summaries) == 9
