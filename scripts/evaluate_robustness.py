r"""Run quantitative robustness evaluation.

Run from the repository root:

    py scripts\evaluate_robustness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.robustness_evaluation import run_robustness_evaluation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate pick-point estimation robustness on degraded synthetic images."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic",
        help="Synthetic dataset directory containing labels.csv and images/.",
    )
    parser.add_argument(
        "--robustness-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "robustness" / "step7",
        help="Directory containing robustness images and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "metrics" / "robustness",
        help="Directory where robustness evaluation reports will be saved.",
    )
    parser.add_argument(
        "--max-source-images",
        type=int,
        default=5,
        help="Number of source images to use if variants must be generated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used if variants must be generated.",
    )
    parser.add_argument(
        "--no-create-variants",
        action="store_true",
        help="Fail instead of creating robustness variants when metadata is missing.",
    )
    return parser.parse_args()


def main() -> int:
    """Run robustness evaluation and print a concise summary."""
    args = parse_args()

    report = run_robustness_evaluation(
        dataset_dir=args.dataset_dir,
        robustness_dir=args.robustness_dir,
        output_dir=args.output_dir,
        create_variants_if_missing=not args.no_create_variants,
        max_source_images=args.max_source_images,
        seed=args.seed,
    )

    print("Robustness evaluation complete.")
    print(f"Variants evaluated: {report.num_variants}")
    print(f"Successful variants: {report.successful_variants}")
    print(f"Success rate: {report.success_rate_percent:.1f}%")
    print(f"Output directory: {report.output_dir}")
    print()
    print("Grouped results:")
    for group in report.group_summaries:
        print(
            f"  {group.transform_name}/{group.severity}: "
            f"success={group.success_rate_percent:.1f}%, "
            f"mean center error={_format_optional(group.mean_center_error_px)} px, "
            f"mean angle error={_format_optional(group.mean_pca_orientation_error_deg)} deg"
        )

    print()
    print("Generated reports:")
    print(f"  Per-variant CSV: {report.per_variant_metrics_csv}")
    print(f"  Summary JSON: {report.summary_json}")
    print(f"  Markdown report: {report.markdown_report}")
    print(f"  Annotated preview grid: {report.preview_grid_path}")

    return 0


def _format_optional(value: float | None) -> str:
    """Format optional float for console output."""
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
