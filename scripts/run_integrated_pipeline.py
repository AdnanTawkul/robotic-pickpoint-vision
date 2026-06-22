r"""Run the integrated YOLO/OpenCV pick-point pipeline.

Run from the repository root:

    py scripts\run_integrated_pipeline.py --input-dir data\sample_images

Improved YOLO attempt:

    py scripts\run_integrated_pipeline.py --input-dir data\sample_images --model yolov8s.pt --confidence 0.10 --img-size 1280 --augment
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.integrated_pipeline import run_integrated_pickpoint_on_folder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run integrated YOLO/OpenCV pick-point estimation."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "sample_images",
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "annotated" / "integrated",
        help="Folder where integrated annotations will be saved.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "metrics" / "integrated_pickpoints.csv",
        help="CSV file for integrated pick-point results.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT / "outputs" / "metrics" / "integrated_pickpoints.json",
        help="JSON file for integrated pick-point results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics YOLO model name or local path.",
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
        "--classes",
        nargs="*",
        default=None,
        help="Optional class-name filter, e.g. --classes bottle cup scissors.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum number of images to process.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=3,
        help="Maximum detections per image to convert into pick targets.",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLO and use OpenCV fallback only.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable OpenCV fallback when YOLO finds no usable target.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the integrated pipeline and print a concise summary."""
    args = parse_args()

    allowed_classes = set(args.classes) if args.classes else None

    image_results, grid_path = run_integrated_pickpoint_on_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metrics_csv=args.metrics_csv,
        summary_json=args.summary_json,
        model_name_or_path=args.model,
        confidence_threshold=args.confidence,
        allowed_class_names=allowed_classes,
        use_yolo=not args.no_yolo,
        fallback_to_opencv=not args.no_fallback,
        max_images=args.max_images,
        max_detections=args.max_detections,
        image_size=args.img_size,
        iou_threshold=args.iou,
        augment=args.augment,
    )

    successful_images = [result for result in image_results if result.success]
    target_count = sum(len(result.results) for result in image_results)
    timings = [result.inference_time_ms for result in image_results]

    print("Integrated pick-point pipeline complete.")
    print(f"Input directory: {args.input_dir}")
    print(f"Images processed: {len(image_results)}")
    print(f"Images with pick targets: {len(successful_images)}")
    print(f"Total pick targets: {target_count}")
    print(f"Mean runtime: {statistics.mean(timings):.3f} ms/image")
    print()
    print("YOLO settings:")
    print(f"  Model: {args.model}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Image size: {args.img_size}")
    print(f"  IoU: {args.iou}")
    print(f"  Augment: {args.augment}")
    print()
    print("Generated files:")
    print(f"  Annotated output directory: {args.output_dir}")
    print(f"  Preview grid: {grid_path}")
    print(f"  Metrics CSV: {args.metrics_csv}")
    print(f"  Summary JSON: {args.summary_json}")

    print()
    print("First results:")
    for image_result in image_results[:5]:
        if not image_result.results:
            print(f"  {image_result.image_name}: no pick target")
            continue

        first = image_result.results[0]
        label = first.detection_class_name or first.method
        print(
            f"  {image_result.image_name}: "
            f"{label}, pick=({first.pick_x:.1f}, {first.pick_y:.1f}), "
            f"angle={first.angle_deg_pca:.1f} deg, method={first.method}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
