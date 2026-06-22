r"""Patch detection.py to make CUDA device handling safe.

Run once from the repository root:

    py scripts\apply_step13e_cuda_patch.py

This replaces resolve_yolo_device() so that forcing cuda:0 will not crash
when PyTorch is installed as CPU-only. It falls back to CPU and prints a clear
warning from CLI scripts via the resolved device display.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DETECTION_PATH = REPO_ROOT / "src" / "pickpoint_vision" / "detection.py"


NEW_FUNCTION = 'def resolve_yolo_device(device: str = "auto") -> str:\n    """Resolve a YOLO inference device safely.\n\n    Returns:\n        - "0" when CUDA is available and GPU was requested/auto-selected\n        - "cpu" when CUDA is unavailable\n        - another explicit non-CUDA device string unchanged\n\n    Ultralytics accepts "0" for the first CUDA GPU and "cpu" for CPU.\n\n    Important:\n        A PC can have an NVIDIA GPU while PyTorch is still CPU-only. In that\n        case, torch.cuda.is_available() is False and YOLO must use CPU until\n        CUDA-enabled PyTorch is installed.\n    """\n    normalized = device.strip().lower()\n\n    try:\n        import torch\n    except ImportError:\n        return "cpu"\n\n    cuda_available = torch.cuda.is_available()\n    cuda_requested = normalized in {"auto", "", "cuda", "cuda:0", "gpu", "0"}\n\n    if normalized in {"auto", ""}:\n        return "0" if cuda_available else "cpu"\n\n    if cuda_requested:\n        return "0" if cuda_available else "cpu"\n\n    return device\n'


def patch_file() -> None:
    """Replace resolve_yolo_device in detection.py."""
    text = DETECTION_PATH.read_text(encoding="utf-8")

    start = text.find("def resolve_yolo_device(")
    if start == -1:
        raise RuntimeError("Could not find resolve_yolo_device() in detection.py")

    end = text.find("@dataclass", start)
    if end == -1:
        raise RuntimeError("Could not find the next @dataclass after resolve_yolo_device()")

    patched = text[:start] + NEW_FUNCTION + "\n\n" + text[end:]
    DETECTION_PATH.write_text(patched, encoding="utf-8")


def main() -> int:
    """Apply the patch."""
    patch_file()
    print("Step 13E CUDA-safe device patch applied successfully.")
    print(f"Updated: {DETECTION_PATH}")
    print()
    print("Now run:")
    print("  py scripts\\check_cuda_setup.py")
    print("  py scripts\\run_yolo_detection.py --input-dir data\\sample_images --device auto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
