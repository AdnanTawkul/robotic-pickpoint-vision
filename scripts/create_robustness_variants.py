r"""Create robustness image variants for qualitative stress testing.

Run from the repository root:

    py scripts\create_robustness_variants.py --clear
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.robustness import create_robustness_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create robustness variants for synthetic images."
    )
    parser.add_argument(
        "--source-images-dir",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic" / "images",
        help="Directory containing source synthetic images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "robustness" / "step7",
        help="Directory where robustness variants will be saved.",
    )
    parser.add_argument(
        "--max-source-images",
        type=int,
        default=5,
        help="Number of source images to transform.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable noise and occlusion.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing robustness images before creating new variants.",
    )
    return parser.parse_args()


def main() -> int:
    """Create robustness variants and print a summary."""
    args = parse_args()

    metadata = create_robustness_dataset(
        source_images_dir=args.source_images_dir,
        output_dir=args.output_dir,
        max_source_images=args.max_source_images,
        seed=args.seed,
        clear_existing=args.clear,
    )

    transform_names = sorted({item.transform_name for item in metadata})

    print("Robustness variant generation complete.")
    print(f"Source images directory: {args.source_images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Source images used: {args.max_source_images}")
    print(f"Variants created: {len(metadata)}")
    print(f"Transforms: {', '.join(transform_names)}")
    print()
    print("Generated files:")
    print(f"  Images: {args.output_dir / 'images'}")
    print(f"  Metadata CSV: {args.output_dir / 'robustness_metadata.csv'}")
    print(f"  Metadata JSON: {args.output_dir / 'robustness_metadata.json'}")
    print(f"  Preview grid: {args.output_dir / 'robustness_preview_grid.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
