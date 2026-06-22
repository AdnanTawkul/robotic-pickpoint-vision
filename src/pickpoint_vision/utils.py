from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_image_files(directory: str | Path) -> list[Path]:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    directory_path = Path(directory)

    if not directory_path.exists():
        return []

    return sorted(
        file_path
        for file_path in directory_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in image_extensions
    )
