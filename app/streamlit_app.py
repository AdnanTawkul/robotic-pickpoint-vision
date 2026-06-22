r"""Streamlit GUI for the pick-point vision project.

Run from the repository root:

    streamlit run app\streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pickpoint_vision.app_utils import (
    integrated_result_to_table_rows,
    parse_class_filter,
    read_image_as_rgb,
    save_uploaded_image_bytes,
    summarize_image_result,
)
from pickpoint_vision.contour_tuning import (
    ContourTuningConfig,
    create_local_background_distance_image,
    estimate_pose_with_contour_tuning,
    segment_with_contour_tuning,
)
from pickpoint_vision.detection import load_yolo_model, resolve_yolo_device
from pickpoint_vision.integrated_pipeline import run_integrated_pickpoint_on_image
from pickpoint_vision.visualization import annotate_pose_result


UPLOAD_DIR = REPO_ROOT / "outputs" / "streamlit" / "uploads"
ANNOTATED_DIR = REPO_ROOT / "outputs" / "streamlit" / "annotated"


DETECTION_PRESETS: dict[str, dict[str, object]] = {
    "Balanced default": {
        "model": "yolov8n.pt",
        "confidence": 0.25,
        "image_size": 640,
        "iou": 0.70,
        "augment": False,
        "class_filter": "",
        "max_detections": 3,
        "note": "Fast first-pass setting for common objects.",
    },
    "Sensitive tabletop": {
        "model": "yolov8s.pt",
        "confidence": 0.10,
        "image_size": 1280,
        "iou": 0.70,
        "augment": True,
        "class_filter": "",
        "max_detections": 5,
        "note": "More sensitive setting for difficult tabletop images. Slower and may add false positives.",
    },
    "High precision": {
        "model": "yolov8n.pt",
        "confidence": 0.80,
        "image_size": 960,
        "iou": 0.55,
        "augment": False,
        "class_filter": "",
        "max_detections": 3,
        "note": "Keeps only high-confidence detections. Useful when you want fewer false positives.",
    },
    "Small objects": {
        "model": "yolov8s.pt",
        "confidence": 0.20,
        "image_size": 1280,
        "iou": 0.65,
        "augment": False,
        "class_filter": "",
        "max_detections": 5,
        "note": "Larger input size helps smaller objects, but increases runtime.",
    },
}


@st.cache_resource(show_spinner=False)
def cached_yolo_model(model_name_or_path: str):
    """Cache YOLO model loading across app interactions."""
    return load_yolo_model(model_name_or_path)


def _display_status_panel(
    targets: int,
    runtime_ms: float,
    mode: str,
    yolo_enabled: bool,
    device_label: str,
) -> None:
    """Display compact status information without oversized truncated text."""
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


def _build_contour_tuning_config() -> tuple[bool, ContourTuningConfig]:
    """Create live contour tuning controls in the sidebar.

    Streamlit reruns the script whenever a slider changes. Because these
    controls are not inside a form, the contour preview updates live.
    """
    st.sidebar.divider()
    st.sidebar.header("Live contour tuning")

    enabled = st.sidebar.checkbox(
        "Enable live contour tuning",
        value=False,
        help=(
            "Updates the OpenCV mask, contour, orientation, and pick point immediately "
            "whenever a slider changes. No extra run button is needed."
        ),
    )

    if not enabled:
        return False, ContourTuningConfig()

    st.sidebar.caption(
        "Live mode: every slider movement reruns only the lightweight OpenCV contour step."
    )

    foreground_sensitivity = st.sidebar.slider(
        "Foreground sensitivity",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help=(
            "Controls how easily pixels become foreground. "
            "Higher values include more pixels; lower values are stricter."
        ),
    )

    blur_kernel_size = st.sidebar.select_slider(
        "Blur size",
        options=[1, 3, 5, 7, 9, 11, 15],
        value=5,
        help="Smooths noise before thresholding. Larger values remove noise but can hide thin details.",
    )

    open_kernel_size = st.sidebar.select_slider(
        "Open kernel",
        options=[1, 3, 5, 7, 9, 11],
        value=5,
        help="Removes small foreground speckles. Larger values are more aggressive.",
    )

    close_kernel_size = st.sidebar.select_slider(
        "Close kernel",
        options=[1, 3, 5, 7, 9, 11, 15],
        value=7,
        help="Fills small holes and gaps inside the object mask.",
    )

    erode_iterations = st.sidebar.slider(
        "Erode iterations",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        help="Shrinks the mask. Useful when the contour is too large.",
    )

    dilate_iterations = st.sidebar.slider(
        "Dilate iterations",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        help="Expands the mask. Useful when the contour is too small.",
    )

    min_contour_area = st.sidebar.slider(
        "Minimum contour area",
        min_value=10,
        max_value=10000,
        value=250,
        step=10,
        help="Filters out tiny false contours.",
    )

    contour_smoothing = st.sidebar.slider(
        "Contour smoothing",
        min_value=0.000,
        max_value=0.050,
        value=0.005,
        step=0.001,
        help="Smooths jagged contour boundaries. Too much smoothing can distort object shape.",
    )

    invert_mask = st.sidebar.checkbox(
        "Invert mask",
        value=False,
        help="Use when the background is selected instead of the object.",
    )

    config = ContourTuningConfig(
        foreground_sensitivity=foreground_sensitivity,
        blur_kernel_size=blur_kernel_size,
        open_kernel_size=open_kernel_size,
        close_kernel_size=close_kernel_size,
        erode_iterations=erode_iterations,
        dilate_iterations=dilate_iterations,
        min_contour_area=min_contour_area,
        contour_smoothing=contour_smoothing,
        invert_mask=invert_mask,
    )

    return True, config


def _pose_to_live_table_row(image_name: str, config: ContourTuningConfig, pose) -> dict[str, object]:
    """Create a table row for the live tuned contour result."""
    return {
        "image_name": image_name,
        "method": "live_contour_tuning",
        "pick_x": round(float(pose.pick_x), 3),
        "pick_y": round(float(pose.pick_y), 3),
        "center_x": round(float(pose.center_x), 3),
        "center_y": round(float(pose.center_y), 3),
        "angle_deg_pca": round(float(pose.angle_deg_pca), 3),
        "angle_deg_min_area_rect": round(float(pose.angle_deg_min_area_rect), 3),
        "contour_area": round(float(pose.contour_area), 3),
        "bbox_x": int(pose.bbox_x),
        "bbox_y": int(pose.bbox_y),
        "bbox_width": int(pose.bbox_width),
        "bbox_height": int(pose.bbox_height),
        "foreground_sensitivity": config.foreground_sensitivity,
        "blur_kernel_size": config.blur_kernel_size,
        "open_kernel_size": config.open_kernel_size,
        "close_kernel_size": config.close_kernel_size,
        "erode_iterations": config.erode_iterations,
        "dilate_iterations": config.dilate_iterations,
        "min_contour_area": config.min_contour_area,
        "contour_smoothing": config.contour_smoothing,
        "invert_mask": config.invert_mask,
    }


def _run_live_contour_tuning_panel(
    input_path: Path,
    image_display_width: int,
    config: ContourTuningConfig,
) -> None:
    """Run and display the live contour tuning panel."""
    st.subheader("Live contour tuning")
    st.caption(
        "Move the sliders in the sidebar. The mask, contour, orientation, and pick point update immediately."
    )

    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error(f"Could not read image: {input_path}")
        return

    try:
        distance_image = create_local_background_distance_image(image_bgr)
        mask = segment_with_contour_tuning(image=image_bgr, config=config)
        tuning_result = estimate_pose_with_contour_tuning(
            image=image_bgr,
            config=config,
            image_name=input_path.name,
        )
        annotated_bgr = annotate_pose_result(
            image=image_bgr,
            mask=mask,
            result=tuning_result.pose_result,
            orientation_source="pca",
        )
    except Exception as exc:
        st.error(f"Live contour tuning failed: {exc}")
        st.info("Try increasing foreground sensitivity, reducing minimum contour area, or toggling invert mask.")
        with st.expander("Current tuning settings"):
            st.json(config.to_dict())
        return

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    distance_rgb = cv2.cvtColor(distance_image, cv2.COLOR_GRAY2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    pose = tuning_result.pose_result
    metric_cols = st.columns(5)
    metric_cols[0].metric("Pick X", f"{pose.pick_x:.1f}")
    metric_cols[1].metric("Pick Y", f"{pose.pick_y:.1f}")
    metric_cols[2].metric("Angle", f"{pose.angle_deg_pca:.1f}°")
    metric_cols[3].metric("Contour area", f"{pose.contour_area:.0f}")
    metric_cols[4].metric("Mask pixels", f"{np.count_nonzero(mask):,}")

    tab_final, tab_debug, tab_data = st.tabs(
        ["Final tuned result", "Mask debug view", "Tuned result data"]
    )

    with tab_final:
        st.image(
            annotated_rgb,
            caption="Live tuned contour + pick point",
            width=image_display_width,
        )

    with tab_debug:
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_rgb, caption="Original image", width=image_display_width)
            st.image(distance_rgb, caption="Local background distance image", width=image_display_width)
        with col2:
            st.image(mask_rgb, caption="Live tuned binary mask", width=image_display_width)
            st.image(annotated_rgb, caption="Live tuned contour + pick point", width=image_display_width)

    with tab_data:
        row = _pose_to_live_table_row(
            image_name=input_path.name,
            config=config,
            pose=pose,
        )
        st.dataframe(pd.DataFrame([row]), width="stretch")
        with st.expander("Current tuning settings JSON"):
            st.json(config.to_dict())

    annotated_output_path = ANNOTATED_DIR / input_path.name.replace(
        input_path.suffix,
        "_live_contour_tuned.png",
    )
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(annotated_output_path), annotated_bgr)

    st.download_button(
        label="Download live tuned annotated image",
        data=annotated_output_path.read_bytes(),
        file_name=annotated_output_path.name,
        mime="image/png",
    )

    csv_data = pd.DataFrame(
        [
            _pose_to_live_table_row(
                image_name=input_path.name,
                config=config,
                pose=pose,
            )
        ]
    ).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download live tuned result CSV",
        data=csv_data,
        file_name="live_contour_tuned_result.csv",
        mime="text/csv",
    )


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Pick-Point Vision Demo",
        page_icon="🎯",
        layout="wide",
    )

    st.title("Vision-Based Pick-Point Estimation")
    st.caption(
        "Upload an image and run either the automatic YOLO/OpenCV pipeline or the live contour tuning workflow."
    )

    with st.sidebar:
        st.header("Pipeline settings")

        mode = st.radio(
            "Pipeline mode",
            options=[
                "YOLO + OpenCV fallback",
                "OpenCV only",
            ],
            index=0,
        )
        use_yolo = mode == "YOLO + OpenCV fallback"

        preset_name = st.selectbox(
            "Detection preset",
            options=list(DETECTION_PRESETS.keys()),
            index=0,
            disabled=not use_yolo,
            help="Choose a general starting point, then fine-tune the settings below.",
        )
        preset = DETECTION_PRESETS[preset_name]

        if use_yolo:
            st.info(str(preset["note"]))

        model_name = st.text_input(
            "YOLO model",
            value=str(preset["model"]),
            disabled=not use_yolo,
            help=(
                "Model weights to use. yolov8n.pt is fastest. yolov8s.pt or yolov8m.pt "
                "can improve detections but are slower."
            ),
        )

        device_option = st.selectbox(
            "YOLO device",
            options=["auto", "cuda:0", "cpu"],
            index=0,
            disabled=not use_yolo,
            help=(
                "auto uses CUDA when PyTorch detects a GPU, otherwise CPU. "
                "Use cuda:0 to request the first NVIDIA GPU, or cpu for debugging."
            ),
        )
        resolved_device = resolve_yolo_device(device_option) if use_yolo else "off"

        confidence = st.slider(
            "YOLO confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(preset["confidence"]),
            step=0.05,
            disabled=not use_yolo,
            help=(
                "Minimum confidence score required to keep a detection. "
                "Lower values find more objects but may add false positives. "
                "Higher values are stricter and may miss objects."
            ),
        )

        image_size = st.select_slider(
            "YOLO image size",
            options=[640, 768, 960, 1280],
            value=int(preset["image_size"]),
            disabled=not use_yolo,
            help=(
                "Input resolution used by YOLO. Larger sizes can improve small-object "
                "detection but increase runtime and GPU memory usage."
            ),
        )

        iou_threshold = st.slider(
            "YOLO IoU threshold",
            min_value=0.30,
            max_value=0.90,
            value=float(preset["iou"]),
            step=0.05,
            disabled=not use_yolo,
            help=(
                "Non-maximum suppression overlap threshold. Lower values remove more "
                "overlapping boxes. Higher values keep more overlapping detections."
            ),
        )

        augment = st.checkbox(
            "Use YOLO test-time augmentation",
            value=bool(preset["augment"]),
            disabled=not use_yolo,
            help=(
                "Runs extra augmented inference passes. It can improve difficult detections, "
                "but it is much slower."
            ),
        )

        class_filter_text = st.text_area(
            "Optional YOLO class filter",
            value=str(preset["class_filter"]),
            help=(
                "Restrict YOLO results to specific class names, separated by commas or new lines. "
                "Examples: bottle, cup, mouse, scissors. Leave empty to allow all classes."
            ),
            disabled=not use_yolo,
        )

        max_detections = st.slider(
            "Maximum detections per image",
            min_value=1,
            max_value=10,
            value=int(preset["max_detections"]),
            step=1,
            disabled=not use_yolo,
            help=(
                "Maximum number of detected boxes that will be converted into pick targets. "
                "Lower values keep the display cleaner."
            ),
        )

        fallback_enabled = st.checkbox(
            "Use OpenCV fallback if YOLO fails",
            value=True,
            disabled=not use_yolo,
            help=(
                "If YOLO finds no usable target, the app tries full-image OpenCV segmentation. "
                "This helps with unknown objects but can be less accurate."
            ),
        )

        if use_yolo:
            st.caption(f"Resolved device: `{resolved_device}`")

        st.divider()
        st.header("Display settings")
        image_display_width = st.slider(
            "Image display width",
            min_value=300,
            max_value=900,
            value=450,
            step=50,
            help="Controls how large the input and annotated images appear in the app.",
        )

    live_tuning_enabled, tuning_config = _build_contour_tuning_config()

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Upload an image to start the demo.")
        st.stop()

    uploaded = save_uploaded_image_bytes(
        image_bytes=uploaded_file.getvalue(),
        original_filename=uploaded_file.name,
        output_dir=UPLOAD_DIR,
    )

    input_path = Path(uploaded.saved_path)
    output_path = ANNOTATED_DIR / input_path.name.replace(input_path.suffix, "_streamlit.png")
    original_rgb = read_image_as_rgb(input_path)

    automatic_tab, live_tuning_tab = st.tabs(
        ["Automatic pipeline", "Live contour tuning"]
    )

    with automatic_tab:
        st.subheader("Automatic YOLO/OpenCV pipeline")
        st.caption(
            "Press the button to run the full pipeline. This path uses YOLO when enabled, then OpenCV pose estimation."
        )

        run_button = st.button("Run automatic pick-point estimation", type="primary")

        if not run_button:
            st.image(original_rgb, caption="Uploaded image", width=image_display_width)
        else:
            class_filter = parse_class_filter(class_filter_text)

            model = None
            if use_yolo:
                with st.spinner("Loading YOLO model..."):
                    model = cached_yolo_model(model_name)

            with st.spinner("Running automatic pick-point pipeline..."):
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

            _display_status_panel(
                targets=int(summary["targets"]),
                runtime_ms=float(summary["runtime_ms"]),
                mode=mode,
                yolo_enabled=use_yolo,
                device_label=resolved_device,
            )

            if use_yolo:
                with st.expander("Detection settings used"):
                    st.write(
                        {
                            "preset": preset_name,
                            "model": model_name,
                            "device_requested": device_option,
                            "device_resolved": resolved_device,
                            "confidence": confidence,
                            "image_size": image_size,
                            "iou_threshold": iou_threshold,
                            "augment": augment,
                            "class_filter": sorted(class_filter) if class_filter else [],
                            "max_detections": max_detections,
                        }
                    )

            left_col, right_col = st.columns(2)

            with left_col:
                st.subheader("Input")
                st.image(original_rgb, caption=uploaded.original_filename, width=image_display_width)

            with right_col:
                st.subheader("Annotated result")
                st.image(annotated_rgb, caption=str(output_path), width=image_display_width)

            st.subheader("Pick-point results")
            if rows:
                dataframe = pd.DataFrame(rows)
                st.dataframe(dataframe, width="stretch")
            else:
                st.warning("No result rows were produced.")

            st.subheader("Download automatic outputs")
            annotated_bytes = output_path.read_bytes()
            st.download_button(
                label="Download automatic annotated image",
                data=annotated_bytes,
                file_name=output_path.name,
                mime="image/png",
            )

            if rows:
                csv_data = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download automatic result CSV",
                    data=csv_data,
                    file_name="automatic_pickpoint_result.csv",
                    mime="text/csv",
                )

            with st.expander("Automatic result JSON"):
                st.json(image_result.to_dict())

    with live_tuning_tab:
        if not live_tuning_enabled:
            st.info("Enable live contour tuning in the sidebar to activate the real-time contour editor.")
            st.image(original_rgb, caption="Uploaded image", width=image_display_width)
        else:
            _run_live_contour_tuning_panel(
                input_path=input_path,
                image_display_width=image_display_width,
                config=tuning_config,
            )


if __name__ == "__main__":
    main()
