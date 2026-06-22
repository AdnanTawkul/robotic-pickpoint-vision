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

from pickpoint_vision.app_utils import (
    integrated_result_to_table_rows,
    parse_class_filter,
    read_image_as_rgb,
    save_uploaded_image_bytes,
    summarize_image_result,
)
from pickpoint_vision.integrated_pipeline import run_integrated_pickpoint_on_image
from pickpoint_vision.detection import load_yolo_model


UPLOAD_DIR = REPO_ROOT / "outputs" / "streamlit" / "uploads"
ANNOTATED_DIR = REPO_ROOT / "outputs" / "streamlit" / "annotated"


@st.cache_resource(show_spinner=False)
def cached_yolo_model(model_name_or_path: str):
    """Cache YOLO model loading across app interactions."""
    return load_yolo_model(model_name_or_path)


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Pick-Point Vision Demo",
        page_icon="🎯",
        layout="wide",
    )

    st.title("Vision-Based Pick-Point Estimation")
    st.caption(
        "Upload an image and run the integrated YOLO/OpenCV pipeline to estimate "
        "object center, orientation, and recommended pick point."
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

        model_name = st.text_input(
            "YOLO model",
            value="yolov8n.pt",
            disabled=not use_yolo,
        )

        confidence = st.slider(
            "YOLO confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            disabled=not use_yolo,
        )

        class_filter_text = st.text_area(
            "Optional YOLO class filter",
            value="",
            help="Example: bottle, cup, scissors",
            disabled=not use_yolo,
        )

        max_detections = st.slider(
            "Maximum detections per image",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            disabled=not use_yolo,
        )

        fallback_enabled = st.checkbox(
            "Use OpenCV fallback if YOLO fails",
            value=True,
            disabled=not use_yolo,
        )

        st.divider()
        st.header("Display settings")
        image_display_width = st.slider(
            "Image display width",
            min_value=300,
            max_value=900,
            value=450,
            step=50,
            help="Reduce this if uploaded phone images appear too large.",
        )

        st.divider()
        st.markdown(
            "**Tip:** The OpenCV path works best when the object has visible "
            "contrast against the background. YOLO works best on common COCO-like objects."
        )

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

    run_button = st.button("Run pick-point estimation", type="primary")

    if not run_button:
        st.image(
            original_rgb,
            caption="Uploaded image",
            width=image_display_width,
        )
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
        )

    annotated_rgb = read_image_as_rgb(output_path)
    summary = summarize_image_result(image_result)
    rows = integrated_result_to_table_rows(image_result)

    if image_result.success:
        st.success(f"Estimated {len(image_result.results)} pick target(s).")
    else:
        st.error(f"No pick target estimated: {image_result.failure_reason}")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Targets", summary["targets"])
    metric_cols[1].metric("Runtime", f"{summary['runtime_ms']:.1f} ms")
    metric_cols[2].metric("Mode", mode)
    metric_cols[3].metric("YOLO", "on" if use_yolo else "off")

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Input")
        st.image(
            original_rgb,
            caption=uploaded.original_filename,
            width=image_display_width,
        )

    with right_col:
        st.subheader("Annotated result")
        st.image(
            annotated_rgb,
            caption=str(output_path),
            width=image_display_width,
        )

    st.subheader("Pick-point results")
    if rows:
        dataframe = pd.DataFrame(rows)
        st.dataframe(dataframe, width="stretch")
    else:
        st.warning("No result rows were produced.")

    st.subheader("Download outputs")
    annotated_bytes = output_path.read_bytes()
    st.download_button(
        label="Download annotated image",
        data=annotated_bytes,
        file_name=output_path.name,
        mime="image/png",
    )

    if rows:
        csv_data = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download result CSV",
            data=csv_data,
            file_name="streamlit_pickpoint_result.csv",
            mime="text/csv",
        )

    with st.expander("Result JSON"):
        st.json(image_result.to_dict())


if __name__ == "__main__":
    main()
