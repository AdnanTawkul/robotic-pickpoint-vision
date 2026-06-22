r"""Create annotated Step 4 visualizations for the synthetic dataset.

Run from the repository root:

    py scripts\visualize_synthetic_results.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.pose_estimation import estimate_pose_from_mask_file
from pickpoint_vision.visualization import annotate_pose_from_files, create_annotation_grid


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate annotated visualizations for synthetic pick-point results."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic",
        help="Synthetic dataset directory containing images/, masks/, and labels.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "annotated" / "step4",
        help="Directory where annotated images will be saved.",
    )
    parser.add_argument(
        "--orientation-source",
        choices=["pca", "min_area_rect"],
        default="pca",
        help="Orientation estimate to visualize.",
    )
    return parser.parse_args()


def load_label_rows(labels_csv: Path) -> list[dict[str, str]]:
    """Load synthetic label rows."""
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Missing labels file: {labels_csv}. "
            "Run: py scripts\\create_synthetic_dataset.py --num-images 20 --clear"
        )

    with labels_csv.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    """Create annotated images and a preview grid."""
    args = parse_args()

    labels_csv = args.dataset_dir / "labels.csv"
    images_dir = args.dataset_dir / "images"
    masks_dir = args.dataset_dir / "masks"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_label_rows(labels_csv)
    if not rows:
        print("No labels found.")
        return 1

    annotated_paths: list[Path] = []

    for row in rows:
        image_name = row["image_name"]
        image_path = images_dir / image_name
        mask_path = masks_dir / image_name.replace(".png", "_mask.png")
        output_path = args.output_dir / image_name.replace(".png", "_annotated.png")

        result = estimate_pose_from_mask_file(mask_path=mask_path, image_name=image_name)

        annotate_pose_from_files(
            image_path=image_path,
            mask_path=mask_path,
            result=result,
            output_path=output_path,
            orientation_source=args.orientation_source,
        )
        annotated_paths.append(output_path)

    grid_path = create_annotation_grid(
        image_paths=annotated_paths,
        output_path=args.output_dir / "annotation_grid.png",
    )

    print("Synthetic visualization complete.")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Annotated images created: {len(annotated_paths)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Preview grid: {grid_path}")
    print()
    print("Open the preview grid to inspect the result:")
    print(f"  {grid_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
