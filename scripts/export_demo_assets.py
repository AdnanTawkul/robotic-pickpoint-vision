r"""Export selected generated outputs into docs/assets for GitHub screenshots.

Run from the repository root after generating outputs:

    py scripts\export_demo_assets.py

Only commit exported assets if they are safe to publish.
"""

from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

ASSET_SOURCES = {
    "synthetic_demo_grid.png": REPO_ROOT / "outputs" / "annotated" / "demo" / "demo_grid.png",
    "robustness_evaluation_grid.png": (
        REPO_ROOT / "outputs" / "metrics" / "robustness" / "robustness_evaluation_grid.png"
    ),
    "integrated_pickpoint_grid.png": (
        REPO_ROOT / "outputs" / "annotated" / "integrated" / "integrated_pickpoint_grid.png"
    ),
}


def export_demo_assets(output_dir: str | Path = REPO_ROOT / "docs" / "assets") -> list[Path]:
    """Copy selected generated demo outputs to docs/assets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_paths: list[Path] = []

    for asset_name, source_path in ASSET_SOURCES.items():
        if not source_path.exists():
            print(f"Skipping missing asset source: {source_path}")
            continue

        target_path = output_dir / asset_name
        shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)
        print(f"Copied: {source_path} -> {target_path}")

    return copied_paths


def main() -> int:
    """Export demo assets and print a short summary."""
    copied_paths = export_demo_assets()

    print()
    print(f"Assets exported: {len(copied_paths)}")
    if copied_paths:
        print("Review these files before committing:")
        for path in copied_paths:
            print(f"  {path}")
    else:
        print("No assets were exported. Generate demo outputs first.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
