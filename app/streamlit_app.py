r"""Streamlit GUI for the pick-point vision project.

Run from the repository root:

    streamlit run app\streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.app_utils import integrated_result_to_table_rows, parse_class_filter, read_image_as_rgb, save_uploaded_image_bytes, summarize_image_result
from pickpoint_vision.detection import load_yolo_model, resolve_yolo_device
from pickpoint_vision.integrated_pipeline import run_integrated_pickpoint_on_image

UPLOAD_DIR = REPO_ROOT / "outputs" / "streamlit" / "uploads"
ANNOTATED_DIR = REPO_ROOT / "outputs" / "streamlit" / "annotated"

DETECTION_PRESETS: dict[str, dict[str, object]] = {
    "Balanced default": {"model": "yolov8n.pt", "confidence": 0.25, "image_size": 640, "iou": 0.70, "augment": False, "class_filter": "", "max_detections": 3, "note": "Fast first-pass setting for common objects."},
    "Sensitive tabletop": {"model": "yolov8s.pt", "confidence": 0.10, "image_size": 1280, "iou": 0.70, "augment": True, "class_filter": "", "max_detections": 5, "note": "More sensitive setting for difficult tabletop images. Slower and may add false positives."},
    "High precision": {"model": "yolov8n.pt", "confidence": 0.80, "image_size": 960, "iou": 0.55, "augment": False, "class_filter": "", "max_detections": 3, "note": "Keeps only high-confidence detections. Useful when you want fewer false positives."},
    "Small objects": {"model": "yolov8s.pt", "confidence": 0.20, "image_size": 1280, "iou": 0.65, "augment": False, "class_filter": "", "max_detections": 5, "note": "Larger input size helps smaller objects, but increases runtime."},
}


@st.cache_resource(show_spinner=False)
def cached_yolo_model(model_name_or_path: str):
    return load_yolo_model(model_name_or_path)


def _display_status_panel(targets: int, runtime_ms: float, mode: str, yolo_enabled: bool, device_label: str) -> None:
    metric_cols = st.columns([1, 1, 1.35, 0.8, 0.9])
    metric_cols[0].metric("Targets", targets)
    metric_cols[1].metric("Runtime", f"{runtime_ms:.1f} ms")
    with metric_cols[2]:
        st.caption("Mode")
        st.markdown(f"**{mode}**")
    with metric_cols[3]:
        st.caption("YOLO")
        st.markdown("**on**" if yolo_enabled else "**off**")
    with metric_cols[4]:
        st.caption("Device")
        st.markdown(f"**{device_label}**")


def main() -> None:
    st.set_page_config(page_title="Pick-Point Vision Demo", page_icon="🎯", layout="wide")
    st.title("Vision-Based Pick-Point Estimation")
    st.caption("Upload an image and run the integrated YOLO/OpenCV pipeline to estimate object center, orientation, and recommended pick point.")

    with st.sidebar:
        st.header("Pipeline settings")
        mode = st.radio("Pipeline mode", options=["YOLO + OpenCV fallback", "OpenCV only"], index=0)
        use_yolo = mode == "YOLO + OpenCV fallback"

        preset_name = st.selectbox("Detection preset", options=list(DETECTION_PRESETS.keys()), index=0, disabled=not use_yolo, help="Choose a general starting point, then fine-tune the settings below.")
        preset = DETECTION_PRESETS[preset_name]
        if use_yolo:
            st.info(str(preset["note"]))

        model_name = st.text_input("YOLO model", value=str(preset["model"]), disabled=not use_yolo, help="Model weights. yolov8n.pt is fastest; yolov8s.pt or yolov8m.pt can improve detections but are slower.")
        device_option = st.selectbox("YOLO device", options=["auto", "cuda:0", "cpu"], index=0, disabled=not use_yolo, help="auto uses CUDA when available, otherwise CPU. Use cuda:0 to force the first NVIDIA GPU.")
        resolved_device = resolve_yolo_device(device_option) if use_yolo else "off"
        confidence = st.slider("YOLO confidence threshold", 0.05, 0.95, float(preset["confidence"]), 0.05, disabled=not use_yolo, help="Minimum confidence score to keep a detection. Lower finds more objects but may add false positives. Higher is stricter but may miss objects.")
        image_size = st.select_slider("YOLO image size", options=[640, 768, 960, 1280], value=int(preset["image_size"]), disabled=not use_yolo, help="Input resolution for YOLO. Larger can improve small-object detection but increases runtime and GPU memory use.")
        iou_threshold = st.slider("YOLO IoU threshold", 0.30, 0.90, float(preset["iou"]), 0.05, disabled=not use_yolo, help="Non-max suppression overlap threshold. Lower removes more overlapping boxes; higher keeps more overlapping detections.")
        augment = st.checkbox("Use YOLO test-time augmentation", value=bool(preset["augment"]), disabled=not use_yolo, help="Runs extra augmented inference passes. It can improve difficult detections, but is much slower.")
        class_filter_text = st.text_area("Optional YOLO class filter", value=str(preset["class_filter"]), help="Restrict YOLO to class names separated by commas or new lines. Examples: bottle, cup, mouse, scissors. Leave empty for all classes.", disabled=not use_yolo)
        max_detections = st.slider("Maximum detections per image", 1, 10, int(preset["max_detections"]), 1, disabled=not use_yolo, help="Maximum detected boxes converted into pick targets. Lower values keep the display cleaner.")
        fallback_enabled = st.checkbox("Use OpenCV fallback if YOLO fails", value=True, disabled=not use_yolo, help="If YOLO finds no usable target, try full-image OpenCV segmentation. Helps with unknown objects but can be less accurate.")
        if use_yolo:
            st.caption(f"Resolved device: `{resolved_device}`")

        st.divider()
        st.header("Display settings")
        image_display_width = st.slider("Image display width", 300, 900, 450, 50, help="Controls how large the input and annotated images appear in the app.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is None:
        st.info("Upload an image to start the demo.")
        st.stop()

    uploaded = save_uploaded_image_bytes(uploaded_file.getvalue(), uploaded_file.name, UPLOAD_DIR)
    input_path = Path(uploaded.saved_path)
    output_path = ANNOTATED_DIR / input_path.name.replace(input_path.suffix, "_streamlit.png")
    original_rgb = read_image_as_rgb(input_path)

    if not st.button("Run pick-point estimation", type="primary"):
        st.image(original_rgb, caption="Uploaded image", width=image_display_width)
        st.stop()

    class_filter = parse_class_filter(class_filter_text)
    model = None
    if use_yolo:
        with st.spinner("Loading YOLO model..."):
            model = cached_yolo_model(model_name)

    with st.spinner("Running integrated pick-point pipeline..."):
        image_result = run_integrated_pickpoint_on_image(
            image_path=input_path,
            output_path=output_path,
            model=model,
            confidence_threshold=confidence,
            allowed_class_names=class_filter,
            use_yolo=use_yolo,
            fallback_to_opencv=fallback_enabled or not use_yolo,
            max_detections=max_detections,
            image_size=image_size,
            iou_threshold=iou_threshold,
            augment=augment,
            device=device_option,
        )

    annotated_rgb = read_image_as_rgb(output_path)
    summary = summarize_image_result(image_result)
    rows = integrated_result_to_table_rows(image_result)

    if image_result.success:
        st.success(f"Estimated {len(image_result.results)} pick target(s).")
    else:
        st.error(f"No pick target estimated: {image_result.failure_reason}")

    _display_status_panel(int(summary["targets"]), float(summary["runtime_ms"]), mode, use_yolo, resolved_device)

    if use_yolo:
        with st.expander("Detection settings used"):
            st.write({"preset": preset_name, "model": model_name, "device_requested": device_option, "device_resolved": resolved_device, "confidence": confidence, "image_size": image_size, "iou_threshold": iou_threshold, "augment": augment, "class_filter": sorted(class_filter) if class_filter else [], "max_detections": max_detections})

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Input")
        st.image(original_rgb, caption=uploaded.original_filename, width=image_display_width)
    with right_col:
        st.subheader("Annotated result")
        st.image(annotated_rgb, caption=str(output_path), width=image_display_width)

    st.subheader("Pick-point results")
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.warning("No result rows were produced.")

    st.subheader("Download outputs")
    st.download_button("Download annotated image", output_path.read_bytes(), output_path.name, "image/png")
    if rows:
        csv_data = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("Download result CSV", csv_data, "streamlit_pickpoint_result.csv", "text/csv")

    with st.expander("Result JSON"):
        st.json(image_result.to_dict())


if __name__ == "__main__":
    main()
