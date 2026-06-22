"""Tests for CUDA device resolution."""

from pickpoint_vision.detection import resolve_yolo_device


def test_cpu_device_resolves_to_cpu() -> None:
    """Explicit CPU should remain CPU."""
    assert resolve_yolo_device("cpu") == "cpu"


def test_auto_device_returns_valid_yolo_device() -> None:
    """Auto should resolve to a valid Ultralytics device value."""
    assert resolve_yolo_device("auto") in {"cpu", "0"}


def test_cuda_request_does_not_return_invalid_value() -> None:
    """Requesting CUDA should not return cuda:0 when PyTorch cannot see CUDA."""
    assert resolve_yolo_device("cuda:0") in {"cpu", "0"}
