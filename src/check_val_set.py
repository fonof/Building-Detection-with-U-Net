"""
Проверка валидационного набора (тот же split 80/20, seed=42, что в create_dataloaders).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import _collect_pairs, _load_binary_mask, _load_rgb_image
from src.utils import ensure_dir, log_ok


def get_val_indices(
    images_dir: Path,
    masks_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> list[int]:
    """Индексы val-части — совпадает с create_dataloaders."""
    image_paths, _ = _collect_pairs(images_dir, masks_dir)
    n = len(image_paths)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    train_size = max(1, int(n * train_ratio))
    if n - train_size == 0:
        train_size = n - 1
    return perm[train_size:]


def check_val_set(
    val_images_dir: str | Path,
    val_masks_dir: str | Path,
    num_samples: int = 10,
    train_ratio: float = 0.8,
    save_path: str | Path = "output/val_set_check.png",
) -> None:
    val_images_dir = Path(val_images_dir)
    val_masks_dir = Path(val_masks_dir)
    save_path = Path(save_path)

    all_images, all_masks = _collect_pairs(val_images_dir, val_masks_dir)
    val_indices = get_val_indices(val_images_dir, val_masks_dir, train_ratio=train_ratio)
    images = [all_images[i] for i in val_indices]
    masks = [all_masks[i] for i in val_indices]

    print(f"Val set: {len(images)} изображений (из {len(all_images)} всего)")

    coverages: list[float] = []
    for i, mask_path in enumerate(masks):
        mask = _load_binary_mask(mask_path)
        coverage = float(mask.mean()) * 100
        coverages.append(coverage)
        if i < num_samples:
            print(f"  Sample {i}: coverage = {coverage:.2f}%")

    arr = np.array(coverages)
    print("\nСтатистика покрытия (весь val set):")
    print(f"  Mean: {arr.mean():.2f}%")
    print(f"  Std:  {arr.std():.2f}%")
    print(f"  Min:  {arr.min():.2f}%")
    print(f"  Max:  {arr.max():.2f}%")

    n_show = min(num_samples, len(images))
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_show):
        img = (_load_rgb_image(images[i]) * 255).astype(np.uint8)
        mask = (_load_binary_mask(masks[i]) * 255).astype(np.uint8)

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title(f"Mask {i}\n({coverages[i]:.1f}%)")
        axes[1, i].axis("off")

    plt.tight_layout()
    ensure_dir(save_path.parent, create=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_ok(f"Сохранено: {save_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    check_val_set(
        PROJECT_ROOT / "data/inria_subset_10k/images",
        PROJECT_ROOT / "data/inria_subset_10k/labels",
        save_path=PROJECT_ROOT / "output/val_set_check.png",
    )
