"""Verify that the Step 1 repository setup is correct.

Run from the repository root:

    py scripts\verify_setup.py
"""

from pathlib import Path
import sys


REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "src/pickpoint_vision/__init__.py",
    "src/pickpoint_vision/utils.py",
    "scripts/verify_setup.py",
    "app",
    "tests",
    "data/sample_images",
    "data/synthetic",
    "outputs/annotated",
    "outputs/metrics",
    "docs/project_plan.md",
    "docs/failure_analysis.md",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    missing_paths = [path for path in REQUIRED_PATHS if not (repo_root / path).exists()]
    if missing_paths:
        print("Step 1 setup verification failed.")
        print("Missing paths:")
        for path in missing_paths:
            print(f"  - {path}")
        return 1

    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))

    try:
        from pickpoint_vision import __version__
        from pickpoint_vision.utils import ensure_directory, get_project_root, list_image_files
    except Exception as exc:
        print("Step 1 setup verification failed.")
        print(f"Could not import package: {exc}")
        return 1

    if __version__ != "0.1.0":
        print("Step 1 setup verification failed.")
        print(f"Unexpected package version: {__version__}")
        return 1

    if get_project_root() != repo_root:
        print("Step 1 setup verification failed.")
        print(f"Project root mismatch: expected {repo_root}, got {get_project_root()}")
        return 1

    ensure_directory(repo_root / "outputs" / "annotated")
    sample_images = list_image_files(repo_root / "data" / "sample_images")

    print("Step 1 setup verification passed.")
    print(f"Repository root: {repo_root}")
    print(f"Sample images currently found: {len(sample_images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
