"""Apply Step 13B integration patch.

Run once from the repository root:

    py scripts\apply_step13b_patch.py

This script updates:
- src/pickpoint_vision/integrated_pipeline.py
- app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRATED_PATH = REPO_ROOT / "src" / "pickpoint_vision" / "integrated_pipeline.py"
STREAMLIT_PATH = REPO_ROOT / "app" / "streamlit_app.py"


def patch_integrated_pipeline() -> None:
    text = INTEGRATED_PATH.read_text(encoding="utf-8")

    import_line = "from pickpoint_vision.real_segmentation import segment_real_object\n"
    if import_line not in text:
        text = text.replace(
            "from pickpoint_vision.pose_estimation import PoseEstimationResult, estimate_pose_from_mask\n",
            "from pickpoint_vision.pose_estimation import PoseEstimationResult, estimate_pose_from_mask\n"
            + import_line,
        )

    # Slightly larger padding gives local background segmentation a better border sample.
    text = text.replace("padding: int = 8", "padding: int = 12")

    replacement = '''def segment_foreground_auto(image: np.ndarray) -> np.ndarray:\n    """Segment foreground using the improved real-object segmenter.\n\n    Kept under the old function name so existing tests and imports continue to work.\n    """\n    return segment_real_object(image)\n\n\n'''

    pattern = (
        r"def segment_foreground_auto\(image: np\.ndarray\) -> np\.ndarray:\n"
        r".*?\n\n(?=def _offset_pose_result)"
    )
    patched, count = re.subn(pattern, replacement, text, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(
            "Could not patch segment_foreground_auto. "
            "Check src/pickpoint_vision/integrated_pipeline.py manually."
        )

    INTEGRATED_PATH.write_text(patched, encoding="utf-8")


def patch_streamlit_app() -> None:
    text = STREAMLIT_PATH.read_text(encoding="utf-8")
    text = text.replace(
        "st.dataframe(dataframe, use_container_width=True)",
        'st.dataframe(dataframe, width="stretch")',
    )
    STREAMLIT_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    patch_integrated_pipeline()
    patch_streamlit_app()
    print("Step 13B patch applied successfully.")
    print(f"Updated: {INTEGRATED_PATH}")
    print(f"Updated: {STREAMLIT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
