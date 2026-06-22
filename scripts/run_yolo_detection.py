r"""Run YOLO object detection on an image folder.

Run from the repository root:

    py scripts\run_yolo_detection.py --input-dir data\synthetic\images --max-images 5

Improved detection attempt:

    py scripts\run_yolo_detection.py --input-dir data\sample_images --model yolov8s.pt --confidence 0.10 --img-size 1280 --augment
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.detection import (
    annotate_detection_file,
    create_detection_preview_grid,
    load_yolo_model,
    run_yolo_on_image,
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
        help="YOLO confidence threshold. Try 0.10 for difficult tabletop images.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="YOLO inference image size. Try 960 or 1280 for small objects.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="YOLO non-max suppression IoU threshold.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable YOLO test-time augmentation. Slower but can improve difficult detections.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Maximum YOLO detections per image before filtering.",
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
    print(f"  Confidence: {args.confidence}")
    print(f"  Image size: {args.img_size}")
    print(f"  IoU: {args.iou}")
    print(f"  Augment: {args.augment}")
    model = load_yolo_model(args.model)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    detections_by_image = {}
    annotated_paths: list[Path] = []
    image_runtimes_ms: list[float] = []

    for image_path in image_paths:
        start_time = time.perf_counter()
        detections = run_yolo_on_image(
            image_path=image_path,
            model=model,
            confidence_threshold=args.confidence,
            allowed_class_names=allowed_classes,
            image_size=args.img_size,
            iou_threshold=args.iou,
            augment=args.augment,
            max_detections=args.max_det,
        )
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        image_runtimes_ms.append(runtime_ms)

        detections_by_image[image_path] = detections

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

    total_detections = sum(len(detections) for detections in detections_by_image.values())
    images_with_detections = sum(bool(detections) for detections in detections_by_image.values())

    print("YOLO detection complete.")
    print(f"Input directory: {args.input_dir}")
    print(f"Images processed: {len(detections_by_image)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Mean runtime: {statistics.mean(image_runtimes_ms):.3f} ms/image")
    print(f"Annotated output directory: {args.output_dir}")
    print(f"Detection CSV: {args.metrics_csv}")
    print(f"Preview grid: {grid_path}")

    print()
    print("Per-image detection summary:")
    for image_path, detections in detections_by_image.items():
        if detections:
            labels = ", ".join(
                f"{detection.class_name}:{detection.confidence:.2f}"
                for detection in detections[:5]
            )
        else:
            labels = "no detections"
        print(f"  {image_path.name}: {labels}")

    if total_detections == 0:
        print()
        print("Note: zero detections can be normal for synthetic rectangles/ellipses.")
        print("For difficult real images, try:")
        print("  --model yolov8s.pt --confidence 0.10 --img-size 1280 --augment")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
