r"""Run the integrated YOLO/OpenCV pick-point pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.detection import resolve_yolo_device
from pickpoint_vision.integrated_pipeline import run_integrated_pickpoint_on_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integrated YOLO/OpenCV pick-point estimation.")
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data" / "sample_images")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "annotated" / "integrated")
    parser.add_argument("--metrics-csv", type=Path, default=REPO_ROOT / "outputs" / "metrics" / "integrated_pickpoints.csv")
    parser.add_argument("--summary-json", type=Path, default=REPO_ROOT / "outputs" / "metrics" / "integrated_pickpoints.json")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--device", type=str, default="auto", help="Use auto, cpu, 0, or cuda:0.")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed_classes = set(args.classes) if args.classes else None
    resolved_device = resolve_yolo_device(args.device)

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
        device=args.device,
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
    print(f"  Device: {args.device} -> {resolved_device}")
    print()
    print("Generated files:")
    print(f"  Preview grid: {grid_path}")
    print(f"  Metrics CSV: {args.metrics_csv}")
    print(f"  Summary JSON: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
