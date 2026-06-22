r"""Run YOLO object detection on an image folder.

Run from the repository root:

    py scripts\run_yolo_detection.py --input-dir data\synthetic\images --max-images 5

For a better real-image demo, put phone images in data\sample_images and run:

    py scripts\run_yolo_detection.py --input-dir data\sample_images
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.detection import (
    annotate_detection_file,
    create_detection_preview_grid,
    load_yolo_model,
    run_yolo_on_folder,
    save_detection_results_csv,
)
from pickpoint_vision.utils import list_image_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO detection and save annotated results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic" / "images",
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "annotated" / "yolo",
        help="Folder where annotated images will be saved.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "metrics" / "yolo_detections.csv",
        help="CSV file where detection results will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model name or local model path.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class-name filter, e.g. --classes bottle cup apple.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum number of images to process.",
    )
    return parser.parse_args()


def main() -> int:
    """Run YOLO detection and save outputs."""
    args = parse_args()

    image_paths = list_image_files(args.input_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        print(f"No images found in: {args.input_dir}")
        print("Add .jpg/.png images or generate synthetic images first:")
        print("  py scripts\\create_synthetic_dataset.py --num-images 20 --clear")
        return 1

    allowed_classes = set(args.classes) if args.classes else None

    print("Loading YOLO model...")
    print(f"  Model: {args.model}")
    model = load_yolo_model(args.model)

    detections_by_image = run_yolo_on_folder(
        input_dir=args.input_dir,
        model=model,
        confidence_threshold=args.confidence,
        allowed_class_names=allowed_classes,
        max_images=args.max_images,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_paths: list[Path] = []

    total_detections = 0
    for image_path, detections in detections_by_image.items():
        total_detections += len(detections)
        output_path = args.output_dir / image_path.name.replace(
            image_path.suffix,
            "_yolo_annotated.png",
        )
        annotate_detection_file(
            image_path=image_path,
            detections=detections,
            output_path=output_path,
        )
        annotated_paths.append(output_path)

    save_detection_results_csv(
        detections_by_image=detections_by_image,
        output_path=args.metrics_csv,
    )
    grid_path = create_detection_preview_grid(
        annotated_image_paths=annotated_paths,
        output_path=args.output_dir / "yolo_detection_grid.png",
    )

    print("YOLO detection complete.")
    print(f"Input directory: {args.input_dir}")
    print(f"Images processed: {len(detections_by_image)}")
    print(f"Total detections: {total_detections}")
    print(f"Annotated output directory: {args.output_dir}")
    print(f"Detection CSV: {args.metrics_csv}")
    print(f"Preview grid: {grid_path}")

    if total_detections == 0:
        print()
        print("Note: zero detections can be normal for synthetic rectangles/ellipses.")
        print("For a better YOLO demo, add phone images of common objects to data\\sample_images.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
