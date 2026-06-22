r"""Run the first end-to-end pick-point vision demo.

Run from the repository root:

    py scripts\run_demo.py --regenerate
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.pipeline import run_synthetic_demo


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the synthetic pick-point estimation demo end to end."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic",
        help="Synthetic dataset directory.",
    )
    parser.add_argument(
        "--annotated-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "annotated" / "demo",
        help="Directory where annotated demo images will be saved.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "metrics" / "demo_summary.json",
        help="Path where the demo JSON summary will be saved.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of synthetic images to generate when dataset regeneration is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when dataset regeneration is enabled.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the synthetic dataset before running the demo.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the demo and print a recruiter-friendly summary."""
    args = parse_args()

    summary = run_synthetic_demo(
        dataset_dir=args.dataset_dir,
        annotated_dir=args.annotated_dir,
        metrics_path=args.metrics_path,
        num_images=args.num_images,
        regenerate_dataset=args.regenerate,
        seed=args.seed,
    )

    print("End-to-end synthetic pick-point demo complete.")
    print(f"Samples processed: {summary.num_samples}")
    print(f"Annotated output directory: {summary.annotated_dir}")
    print(f"Preview grid: {summary.preview_grid_path}")
    print(f"Metrics JSON: {summary.metrics_path}")
    print()
    print("Demo summary:")
    print(f"  Mean center error: {summary.mean_center_error_px:.3f} px")
    print(f"  Max center error: {summary.max_center_error_px:.3f} px")
    print(f"  Mean PCA orientation error: {summary.mean_pca_orientation_error_deg:.3f} deg")
    print(
        "  Mean minAreaRect orientation error: "
        f"{summary.mean_min_area_rect_orientation_error_deg:.3f} deg"
    )
    print(f"  Mean pose-estimation time: {summary.mean_inference_time_ms:.3f} ms/image")
    print()
    print("Open this image to inspect the demo visually:")
    print(f"  {summary.preview_grid_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
