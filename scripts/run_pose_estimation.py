r"""Run Step 3 pose estimation on the synthetic dataset.

Run from the repository root:

    py scripts\run_pose_estimation.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.pose_estimation import (
    axis_orientation_error_deg,
    estimate_pose_from_mask_file,
    save_pose_results_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Estimate object center and orientation from synthetic dataset masks."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic",
        help="Synthetic dataset directory containing labels.csv and masks/.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "metrics" / "pose_estimation_step3.csv",
        help="Where to save pose-estimation results.",
    )
    return parser.parse_args()


def load_ground_truth_rows(labels_csv: Path) -> list[dict[str, str]]:
    """Load synthetic labels from CSV."""
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Missing labels file: {labels_csv}. "
            "Run: py scripts\\create_synthetic_dataset.py --num-images 20 --clear"
        )

    with labels_csv.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    """Run pose estimation and print a concise accuracy summary."""
    args = parse_args()

    labels_csv = args.dataset_dir / "labels.csv"
    masks_dir = args.dataset_dir / "masks"

    rows = load_ground_truth_rows(labels_csv)
    if not rows:
        print("No labels found.")
        return 1

    results = []
    center_errors = []
    pca_angle_errors = []
    rect_angle_errors = []

    for row in rows:
        image_name = row["image_name"]
        mask_name = image_name.replace(".png", "_mask.png")
        mask_path = masks_dir / mask_name

        result = estimate_pose_from_mask_file(mask_path=mask_path, image_name=image_name)
        results.append(result)

        gt_center_x = float(row["center_x"])
        gt_center_y = float(row["center_y"])
        gt_angle_deg = float(row["angle_deg"])

        center_error = ((result.center_x - gt_center_x) ** 2 + (result.center_y - gt_center_y) ** 2) ** 0.5
        pca_angle_error = axis_orientation_error_deg(result.angle_deg_pca, gt_angle_deg)
        rect_angle_error = axis_orientation_error_deg(result.angle_deg_min_area_rect, gt_angle_deg)

        center_errors.append(center_error)
        pca_angle_errors.append(pca_angle_error)
        rect_angle_errors.append(rect_angle_error)

    save_pose_results_csv(results=results, output_path=args.output_csv)

    print("Pose estimation demo complete.")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Samples evaluated: {len(results)}")
    print(f"Results CSV: {args.output_csv}")
    print()
    print("Accuracy summary against synthetic ground truth:")
    print(f"  Mean center error: {statistics.mean(center_errors):.3f} px")
    print(f"  Max center error: {max(center_errors):.3f} px")
    print(f"  Mean PCA orientation error: {statistics.mean(pca_angle_errors):.3f} deg")
    print(f"  Mean minAreaRect orientation error: {statistics.mean(rect_angle_errors):.3f} deg")
    print()
    print("First 5 estimates:")
    for result, center_error, pca_error in zip(results[:5], center_errors[:5], pca_angle_errors[:5]):
        print(
            f"  {result.image_name}: "
            f"center=({result.center_x:.1f}, {result.center_y:.1f}), "
            f"PCA angle={result.angle_deg_pca:.1f} deg, "
            f"center error={center_error:.2f} px, "
            f"PCA angle error={pca_error:.2f} deg"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
