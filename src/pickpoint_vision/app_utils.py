"""Helper utilities for the Streamlit demo app."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import cv2
import numpy as np

from pickpoint_vision.integrated_pipeline import IntegratedImageResult


@dataclass(frozen=True)
class SavedUpload:
    """Information about a saved uploaded image."""

    original_filename: str
    safe_filename: str
    saved_path: str

    def to_dict(self) -> dict[str, str]:
        """Convert upload information to a dictionary."""
        return asdict(self)


def sanitize_filename(filename: str) -> str:
    """Return a safe filename for local demo output."""
    filename = filename.strip().replace("\\", "_").replace("/", "_")
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
    filename = filename.strip("._")

    if not filename:
        return "uploaded_image.png"

    return filename


def save_uploaded_image_bytes(
    image_bytes: bytes,
    original_filename: str,
    output_dir: str | Path,
) -> SavedUpload:
    """Save uploaded image bytes to disk.

    Streamlit gives uploaded files as bytes. Saving them to disk lets the same
    command-line pipeline code process app uploads.
    """
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = sanitize_filename(original_filename)
    saved_path = output_dir / safe_filename
    saved_path.write_bytes(image_bytes)

    return SavedUpload(
        original_filename=original_filename,
        safe_filename=safe_filename,
        saved_path=str(saved_path),
    )


def read_image_as_rgb(image_path: str | Path) -> np.ndarray:
    """Load an image with OpenCV and convert it to RGB for Streamlit display."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def parse_class_filter(class_text: str) -> set[str] | None:
    """Parse a comma/space separated class filter string."""
    cleaned = class_text.strip()
    if not cleaned:
        return None

    parts = re.split(r"[,;\n]+|\s{2,}", cleaned)
    classes = {part.strip().lower() for part in parts if part.strip()}

    return classes or None


def integrated_result_to_table_rows(image_result: IntegratedImageResult) -> list[dict[str, object]]:
    """Convert integrated pick-point results into Streamlit-friendly rows."""
    rows: list[dict[str, object]] = []

    for index, result in enumerate(image_result.results, start=1):
        rows.append(
            {
                "target": index,
                "method": result.method,
                "class": result.detection_class_name or "n/a",
                "confidence": result.detection_confidence,
                "pick_x": result.pick_x,
                "pick_y": result.pick_y,
                "center_x": result.center_x,
                "center_y": result.center_y,
                "angle_deg": result.angle_deg_pca,
                "bbox_x": result.bbox_x,
                "bbox_y": result.bbox_y,
                "bbox_width": result.bbox_width,
                "bbox_height": result.bbox_height,
                "runtime_ms": result.inference_time_ms,
            }
        )

    return rows


def summarize_image_result(image_result: IntegratedImageResult) -> dict[str, object]:
    """Create a compact summary for the app."""
    return {
        "image_name": image_result.image_name,
        "success": image_result.success,
        "targets": len(image_result.results),
        "runtime_ms": image_result.inference_time_ms,
        "failure_reason": image_result.failure_reason,
        "annotated_image_path": image_result.annotated_image_path,
    }
