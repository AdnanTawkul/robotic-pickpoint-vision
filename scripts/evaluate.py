r"""Evaluate the current synthetic pick-point estimation baseline.

Run from the repository root:

    py scripts\evaluate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.evaluation import run_synthetic_evaluation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic pick-point center and orientation estimation."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic",
        help="Synthetic dataset directory containing labels.csv and masks/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "metrics" / "evaluation",
        help="Directory where evaluation reports will be saved.",
    )
    parser.add_argument(
        "--center-threshold-px",
        type=float,
        default=2.0,
        help="Pass threshold for center-point error in pixels.",
    )
    parser.add_argument(
        "--orientation-threshold-deg",
        type=float,
        default=5.0,
        help="Pass threshold for PCA orientation error in degrees.",
    )
    return parser.parse_args()


def main() -> int:
    """Run evaluation and print a concise summary."""
    args = parse_args()

    report = run_synthetic_evaluation(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        center_error_pass_threshold_px=args.center_threshold_px,
        orientation_error_pass_threshold_deg=args.orientation_threshold_deg,
    )

    print("Evaluation complete.")
    print(f"Samples evaluated: {report.num_samples}")
    print(f"Output directory: {report.output_dir}")
    print()
    print("Portfolio metrics:")
    print(f"  Mean center error: {report.center_error_px.mean:.3f} px")
    print(f"  Median center error: {report.center_error_px.median:.3f} px")
    print(f"  P90 center error: {report.center_error_px.p90:.3f} px")
    print(f"  Max center error: {report.center_error_px.max:.3f} px")
    print(f"  Mean PCA orientation error: {report.pca_orientation_error_deg.mean:.3f} deg")
    print(f"  P90 PCA orientation error: {report.pca_orientation_error_deg.p90:.3f} deg")
    print(f"  Mean minAreaRect orientation error: {report.min_area_rect_orientation_error_deg.mean:.3f} deg")
    print(f"  Mean inference time: {report.inference_time_ms.mean:.3f} ms/image")
    print()
    print("Pass-rate checks:")
    print(
        f"  Center error <= {report.center_error_pass_threshold_px:.1f} px: "
        f"{report.center_pass_rate:.1f}%"
    )
    print(
        f"  PCA orientation error <= {report.orientation_error_pass_threshold_deg:.1f} deg: "
        f"{report.pca_orientation_pass_rate:.1f}%"
    )
    print()
    print("Generated reports:")
    print(f"  Per-sample CSV: {report.per_sample_metrics_csv}")
    print(f"  Summary JSON: {report.summary_json}")
    print(f"  Markdown report: {report.markdown_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
