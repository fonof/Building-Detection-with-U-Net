"""
Подготовка Inria Aerial Image Labeling Dataset.
Нарезка больших изображений (5000x5000) на патчи 256x256.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def crop_to_patches(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    patch_size: int = 256,
    stride: int = 128,
) -> int:
    """
    Нарезает большое изображение на перекрывающиеся патчи.

    Args:
        image_path: путь к изображению
        mask_path: путь к маске
        output_dir: папка для сохранения
        patch_size: размер патча (256)
        stride: шаг (128 = 50% перекрытие)
    """
    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"❌ Не удалось загрузить: {image_path.name}")
        return 0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    count = 0

    output_images = output_dir / "images"
    output_masks = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    existing = len(list(output_images.glob("*.png")))

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_img = img[y : y + patch_size, x : x + patch_size]
            patch_mask = mask[y : y + patch_size, x : x + patch_size]

            # Пропускаем пустые маски (меньше 1% зданий)
            if patch_mask.mean() < 2.55:  # 1% от 255
                continue

            idx = existing + count
            cv2.imwrite(
                str(output_images / f"{idx:05d}.png"),
                cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(str(output_masks / f"{idx:05d}.png"), patch_mask)
            count += 1

    return count


def _find_image_mask_dirs(source: Path) -> tuple[Path | None, Path | None]:
    """
    Ищет папки images/ и labels|gt/ в разных вариантах структуры Inria.

    Поддерживает:
    - Inria_dataset_train/images + labels
    - train/images + train/gt  (официальный NEW2)
    - data/train/images + data/train/gt  (HuggingFace)
    """
    candidates = [
        source,
        source / "Inria_dataset_train",
        source / "AerialImageDataset" / "train",
        source / "NEW2-AerialImageDataset" / "AerialImageDataset" / "train",
        source / "NEW2-AerialImageDataset" / "train",
        source / "train",
        source / "data" / "train",
    ]
    for base in candidates:
        if not base.exists():
            continue

        images_dir = base / "images"
        if not images_dir.is_dir():
            continue

        for mask_name in ("labels", "gt"):
            masks_dir = base / mask_name
            if masks_dir.is_dir():
                return images_dir, masks_dir

    # Плоская структура: *.tif в одной папке
    tifs = list(source.glob("*.tif")) + list(source.glob("*.tiff"))
    if tifs:
        return source, source

    return None, None


def prepare_inria(
    source_dir: str | Path,
    output_dir: str | Path,
    patch_size: int = 256,
    stride: int = 128,
) -> int:
    """Подготовка всего датасета Inria."""
    source = Path(source_dir)
    output = Path(output_dir)

    print("=" * 60)
    print("ПОДГОТОВКА INRIA AERIAL IMAGE LABELING DATASET")
    print("=" * 60)

    images_dir, masks_dir = _find_image_mask_dirs(source)

    if images_dir is None:
        print("❌ Не найдены изображения (.tif файлы)")
        print("   Сначала: python src/download_inria.py")
        return 0

    print(f"\n📁 Источник: {images_dir}")
    print(f"📁 Маски: {masks_dir}")

    image_files = sorted(images_dir.glob("*.tif")) + sorted(images_dir.glob("*.tiff"))
    print(f"\n📊 Найдено больших изображений: {len(image_files)}")

    total_patches = 0
    city_stats: dict[str, int] = {}

    print(f"\n✂️  Нарезка на патчи {patch_size}x{patch_size} (stride={stride})")
    print("-" * 60)

    for img_path in tqdm(image_files, desc="Обработка"):
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            mask_path = masks_dir / f"{img_path.stem}_label.tif"
        if not mask_path.exists():
            mask_path = masks_dir / f"{img_path.stem}_label.tiff"
        if not mask_path.exists():
            print(f"⚠️  Нет маски для {img_path.name}")
            continue

        patches = crop_to_patches(
            img_path, mask_path, output, patch_size=patch_size, stride=stride
        )

        # Группируем по городу (austin1 -> austin)
        city_name = img_path.stem.rstrip("0123456789") or img_path.stem
        city_stats[city_name] = city_stats.get(city_name, 0) + patches
        total_patches += patches

        print(f"✅ {img_path.name}: {patches} патчей")

    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)

    for city, count in sorted(city_stats.items()):
        print(f"  {city:15s}: {count:5d} патчей")

    print("-" * 60)
    print(f"🎉 ВСЕГО ПАТЧЕЙ: {total_patches}")
    print(f"📁 Сохранено в: {output}")

    final_images = len(list((output / "images").glob("*.png")))
    final_masks = len(list((output / "labels").glob("*.png")))

    print(f"\n✅ Проверка:")
    print(f"   Images: {final_images}")
    print(f"   Masks: {final_masks}")

    if final_images == final_masks == total_patches:
        print("   ✅ Все совпадает!")
    else:
        print("   ⚠️  Есть расхождения!")

    return total_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/Inria_dataset_train",
        help="Папка с исходными данными",
    )
    parser.add_argument(
        "--output",
        default="data/inria_patches",
        help="Папка для сохранения патчей",
    )
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    source = PROJECT_ROOT / args.source
    output = PROJECT_ROOT / args.output

    if not source.exists():
        # Пробуем альтернативные пути после official download
        for alt in (
            PROJECT_ROOT / "data" / "official_extracted" / "NEW2-AerialImageDataset" / "train",
            PROJECT_ROOT / "data" / "train",
        ):
            if alt.exists():
                source = alt
                break

    prepare_inria(source, output, args.patch_size, args.stride)
    return 0


if __name__ == "__main__":
    sys.exit(main())
