"""Tests for the end-to-end demo pipeline."""

from pathlib import Path

from pickpoint_vision.pipeline import run_synthetic_demo


def test_run_synthetic_demo(tmp_path: Path) -> None:
    """The demo pipeline should create annotations and a metrics summary."""
    dataset_dir = tmp_path / "synthetic"
    annotated_dir = tmp_path / "annotated"
    metrics_path = tmp_path / "metrics" / "demo_summary.json"

    summary = run_synthetic_demo(
        dataset_dir=dataset_dir,
        annotated_dir=annotated_dir,
        metrics_path=metrics_path,
        num_images=4,
        regenerate_dataset=True,
        seed=123,
    )

    assert summary.num_samples == 4
    assert metrics_path.exists()
    assert Path(summary.preview_grid_path).exists()
    assert summary.mean_center_error_px < 2.0
    assert summary.mean_pca_orientation_error_deg < 5.0

    annotated_images = sorted(annotated_dir.glob("*_demo_annotated.png"))
    assert len(annotated_images) == 4
