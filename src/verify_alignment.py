"""
Проверка выравнивания изображений и масок.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, log_error, log_ok, log_warn


def _collect_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted(images_dir.glob("*.png"))
    mask_by_stem = {p.stem: p for p in masks_dir.glob("*.png")}

    pairs: list[tuple[Path, Path]] = []
    for img_path in images:
        mask_path = mask_by_stem.get(img_path.stem)
        if mask_path is not None:
            pairs.append((img_path, mask_path))
    return pairs


def verify_single_pair(
    img_path: Path,
    mask_path: Path,
    output_path: Path,
) -> bool:
    """Проверяет один патч: накладывает маску на изображение."""
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        log_error(f"Не удалось загрузить: {img_path.name}")
        return False

    if img.shape[:2] != mask.shape[:2]:
        log_error("РАЗМЕРЫ НЕ СОВПАДАЮТ!")
        print(f"   Image: {img.shape[:2]}")
        print(f"   Mask:  {mask.shape[:2]}")
        return False

    print(f"✅ Размеры совпадают: {img.shape[:2]}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_colored = np.zeros_like(img_rgb)
    mask_colored[mask > 128] = [255, 0, 0]

    overlay = img_rgb.copy()
    cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0, overlay)

    coverage = (mask > 128).sum() / mask.size * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Mask (coverage: {coverage:.1f}%)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (red=buildings)")
    axes[2].axis("off")

    plt.tight_layout()
    ensure_dir(output_path.parent, create=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_ok(f"Сохранено: {output_path}")
    return True


def scan_all_pairs(pairs: list[tuple[Path, Path]]) -> dict[str, int]:
    """Быстрая проверка всех пар на несовпадение размеров."""
    stats = {"ok": 0, "size_mismatch": 0, "load_error": 0}
    mismatches: list[str] = []

    for img_path, mask_path in pairs:
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            stats["load_error"] += 1
            continue
        if img.shape[:2] != mask.shape[:2]:
            stats["size_mismatch"] += 1
            if len(mismatches) < 5:
                mismatches.append(
                    f"{img_path.name}: img={img.shape[:2]} mask={mask.shape[:2]}"
                )
        else:
            stats["ok"] += 1

    print(f"\nСканирование {len(pairs)} пар:")
    print(f"  ✅ OK:              {stats['ok']}")
    print(f"  ❌ Size mismatch:   {stats['size_mismatch']}")
    print(f"  ❌ Load error:      {stats['load_error']}")
    for m in mismatches:
        print(f"     - {m}")

    return stats


def check_random_samples(
    images_dir: str | Path,
    masks_dir: str | Path,
    num_samples: int = 5,
    output_dir: str | Path = "output",
    seed: int = 42,
) -> None:
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    all_images = sorted(images_dir.glob("*.png"))
    all_masks = sorted(masks_dir.glob("*.png"))
    pairs = _collect_pairs(images_dir, masks_dir)

    print(f"Images: {len(all_images)}, Masks: {len(all_masks)}, Paired: {len(pairs)}")
    if len(all_images) != len(all_masks):
        log_warn("Количество images и masks не совпадает!")
    if len(pairs) < len(all_images):
        log_warn(f"Нет пары по stem для {len(all_images) - len(pairs)} изображений")

    scan_all_pairs(pairs)

    random.seed(seed)
    selected = random.sample(pairs, min(num_samples, len(pairs)))

    for i, (img_path, mask_path) in enumerate(selected):
        print(f"\n--- Sample {i} ({img_path.stem}) ---")
        print(f"Image: {img_path.name}")
        print(f"Mask:  {mask_path.name}")
        verify_single_pair(
            img_path,
            mask_path,
            output_dir / f"alignment_check_{img_path.stem}.png",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", default="data/inria_subset_10k/images")
    p.add_argument("--masks_dir", default="data/inria_subset_10k/labels")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--output_dir", default="output")
    return p.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    check_random_samples(
        PROJECT_ROOT / args.images_dir,
        PROJECT_ROOT / args.masks_dir,
        num_samples=args.num_samples,
        output_dir=PROJECT_ROOT / args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
