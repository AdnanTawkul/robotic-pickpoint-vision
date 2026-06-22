r"""Create a synthetic dataset for pick-point estimation.

Run from the repository root:

    py scripts\create_synthetic_dataset.py --num-images 20 --clear
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allows the script to run before the package is installed with `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.synthetic_data import create_synthetic_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic rotated-object images with ground-truth pick-point labels."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic",
        help="Directory where the generated dataset will be saved.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of synthetic images to generate.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Generated image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Generated image height in pixels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset generation.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing generated images and masks before creating the new dataset.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the dataset and print a short summary."""
    args = parse_args()

    labels = create_synthetic_dataset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        image_width=args.width,
        image_height=args.height,
        seed=args.seed,
        clear_existing=args.clear,
    )

    print("Synthetic dataset generation complete.")
    print(f"Output directory: {args.output_dir}")
    print(f"Images created: {len(labels)}")
    print(f"Labels CSV: {args.output_dir / 'labels.csv'}")
    print(f"Labels JSON: {args.output_dir / 'labels.json'}")
    print(f"Preview grid: {args.output_dir / 'preview_grid.png'}")

    first_label = labels[0]
    print()
    print("First generated label:")
    print(
        f"  image={first_label.image_name}, "
        f"shape={first_label.shape}, "
        f"center=({first_label.center_x:.1f}, {first_label.center_y:.1f}), "
        f"angle={first_label.angle_deg:.1f} deg"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
