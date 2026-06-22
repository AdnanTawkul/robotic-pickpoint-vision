r"""Apply Step 13D CUDA/device and Streamlit UI patch.

Run from the repository root:

    py scripts\apply_step13d_patch.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def replace_file(relative_path: str, content: str) -> None:
    path = REPO_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Updated: {path}")


def main() -> int:
    # The full updated files are included in this ZIP. This script is kept for users who prefer
    # an explicit patch command in GitHub Desktop workflows.
    print("Step 13D files are already present after merging this ZIP.")
    print("No extra patch action is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
