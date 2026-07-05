"""
Скачивание Inria Aerial Image Labeling Dataset.

Источники (по приоритету):
1. https://download.inria.fr/.../Inria_dataset_train.zip  (8 GB, часто недоступен)
2. https://files.inria.fr/aerialimagelabeling/  (официальные 7z, ~21 GB)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, log_error, log_info, log_ok, log_warn

# Kaggle-style repack (8 GB) — DNS download.inria.fr часто не резолвится
ZIP_URL = "https://download.inria.fr/AerialImageLabeling/Inria_dataset_train.zip"

# Официальный источник (5 частей по 4 GB)
OFFICIAL_PARTS = [
    f"https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.{i:03d}"
    for i in range(1, 6)
]


def _curl_download(url: str, dest: Path) -> bool:
    """Скачивание через curl.exe (Windows)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log_info(f"Уже есть: {dest.name}")
        return True

    log_info(f"Скачивание: {url}")
    log_info(f"Куда: {dest}")
    cmd = [
        "curl.exe",
        "-L",
        "--fail",
        "--continue-at",
        "-",
        "-o",
        str(dest),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
        return dest.exists() and dest.stat().st_size > 0
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log_warn(f"curl не удался: {exc}")
        return False


def download_zip(data_dir: Path) -> Path | None:
    """Пробует скачать Inria_dataset_train.zip."""
    archive = data_dir / "Inria_dataset_train.zip"
    if _curl_download(ZIP_URL, archive):
        return archive
    log_warn(f"ZIP недоступен: {ZIP_URL}")
    return None


def download_official_7z(data_dir: Path) -> list[Path]:
    """Скачивает официальные 7z-части с files.inria.fr."""
    parts_dir = ensure_dir(data_dir / "official", create=True)
    downloaded: list[Path] = []

    for url in OFFICIAL_PARTS:
        part = parts_dir / Path(url).name
        if _curl_download(url, part):
            downloaded.append(part)
        else:
            log_error(f"Не удалось скачать: {part.name}")
            break

    return downloaded


def extract_zip(archive: Path, dest: Path) -> Path | None:
    """Распаковывает zip."""
    log_info(f"Распаковка ZIP: {archive.name}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)
    log_ok(f"Распаковано в {dest}")
    return dest


def extract_7z(first_part: Path, dest: Path) -> Path | None:
    """Распаковывает split 7z через 7-Zip (Windows) или py7zr."""
    log_info(f"Распаковка 7z: {first_part.name} (может занять 10-20 мин)")
    dest.mkdir(parents=True, exist_ok=True)

    seven_zip = Path(r"C:\Program Files\7-Zip\7z.exe")
    if not seven_zip.exists():
        seven_zip = Path(r"C:\Program Files (x86)\7-Zip\7z.exe")

    if seven_zip.exists():
        import subprocess

        subprocess.run(
            [str(seven_zip), "x", str(first_part), f"-o{dest}", "-y"],
            check=True,
        )
    else:
        try:
            import py7zr
        except ImportError:
            log_error("Установите 7-Zip или py7zr")
            return None
        with py7zr.SevenZipFile(first_part, mode="r") as archive:
            archive.extractall(path=dest)

    inner_zip = dest / "NEW2-AerialImageDataset.zip"
    if inner_zip.exists():
        log_info("Распаковка NEW2-AerialImageDataset.zip...")
        if seven_zip.exists():
            import subprocess

            inner_dest = dest / "NEW2-AerialImageDataset"
            subprocess.run(
                [str(seven_zip), "x", str(inner_zip), f"-o{inner_dest}", "-y"],
                check=True,
            )
        else:
            with zipfile.ZipFile(inner_zip, "r") as zf:
                zf.extractall(dest / "NEW2-AerialImageDataset")
        return dest / "NEW2-AerialImageDataset"

    return dest


def find_train_root(base: Path) -> Path | None:
    """Ищет папку train/images после распаковки."""
    candidates = [
        base / "Inria_dataset_train",
        base / "NEW2-AerialImageDataset" / "train",
        base / "train",
        base,
    ]
    for path in candidates:
        if (path / "images").is_dir() and (
            (path / "labels").is_dir() or (path / "gt").is_dir()
        ):
            return path
        # официальная структура: train/images + train/gt на уровень выше
        if path.name == "train" and (path / "images").is_dir():
            return path
    # NEW2: корень содержит train/
    train = base / "train"
    if (train / "images").is_dir():
        return train
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Inria Aerial Dataset")
    parser.add_argument(
        "--method",
        choices=["auto", "zip", "official"],
        default="auto",
        help="auto: zip, затем official",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--skip_extract", action="store_true")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    data_dir = ensure_dir(PROJECT_ROOT / args.data_dir, create=True)

    print("⏳ Скачивание Inria dataset (8-21 GB, 10-60 мин)...")

    extracted: Path | None = None

    if args.method in ("auto", "zip"):
        archive = download_zip(data_dir)
        if archive and not args.skip_extract:
            extracted = extract_zip(archive, data_dir)

    if extracted is None and args.method in ("auto", "official"):
        parts = download_official_7z(data_dir)
        if len(parts) == 5 and not args.skip_extract:
            extracted = extract_7z(parts[0], data_dir / "official_extracted")

    train_root = find_train_root(data_dir) if extracted is None else find_train_root(extracted)
    if train_root is None and extracted:
        train_root = find_train_root(extracted.parent)

    if train_root:
        log_ok(f"Train data: {train_root}")
        log_info("Запустите: python src/prepare_inria.py")
        return 0

    if extracted or list(data_dir.glob("**/*.tif")):
        log_warn("Данные скачаны, но структура не распознана. Проверьте data/")
        return 0

    log_error(
        "Скачивание не удалось. Альтернативы:\n"
        "  1. Kaggle: kaggle datasets download -d thedevastator/inria-aerial-image-labeling-dataset\n"
        "  2. Вручную: https://project.inria.fr/aerialimagelabeling/files/\n"
        "  3. HuggingFace: blanchon/INRIA-Aerial-Image-Labeling"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
