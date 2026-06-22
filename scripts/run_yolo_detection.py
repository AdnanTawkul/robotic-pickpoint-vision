r"""Run YOLO object detection on an image folder."""

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

from pickpoint_vision.detection import annotate_detection_file, create_detection_preview_grid, load_yolo_model, resolve_yolo_device, run_yolo_on_image, save_detection_results_csv
from pickpoint_vision.utils import list_image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detection and save annotated results.")
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data" / "synthetic" / "images")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "annotated" / "yolo")
    parser.add_argument("--metrics-csv", type=Path, default=REPO_ROOT / "outputs" / "metrics" / "yolo_detections.csv")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", help="Use auto, cpu, 0, or cuda:0.")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--max-images", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = list_image_files(args.input_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        print(f"No images found in: {args.input_dir}")
        return 1

    allowed_classes = set(args.classes) if args.classes else None
    resolved_device = resolve_yolo_device(args.device)

    print("Loading YOLO model...")
    print(f"  Model: {args.model}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Image size: {args.img_size}")
    print(f"  IoU: {args.iou}")
    print(f"  Augment: {args.augment}")
    print(f"  Device: {args.device} -> {resolved_device}")
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
            device=args.device,
        )
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        image_runtimes_ms.append(runtime_ms)
        detections_by_image[image_path] = detections
        output_path = args.output_dir / image_path.name.replace(image_path.suffix, "_yolo_annotated.png")
        annotate_detection_file(image_path, detections, output_path)
        annotated_paths.append(output_path)

    save_detection_results_csv(detections_by_image, args.metrics_csv)
    grid_path = create_detection_preview_grid(annotated_paths, args.output_dir / "yolo_detection_grid.png")

    total_detections = sum(len(detections) for detections in detections_by_image.values())
    images_with_detections = sum(bool(detections) for detections in detections_by_image.values())
    print("YOLO detection complete.")
    print(f"Images processed: {len(detections_by_image)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Mean runtime: {statistics.mean(image_runtimes_ms):.3f} ms/image")
    print(f"Preview grid: {grid_path}")
    print()
    print("Per-image detection summary:")
    for image_path, detections in detections_by_image.items():
        labels = ", ".join(f"{d.class_name}:{d.confidence:.2f}" for d in detections[:5]) if detections else "no detections"
        print(f"  {image_path.name}: {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
