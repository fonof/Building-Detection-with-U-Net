"""
Создание подмножества Inria для быстрого тестирования
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_subset(
    source_dir: str | Path,
    output_dir: str | Path,
    num_samples: int = 1000,
    seed: int = 42,
) -> int:
    """
    Создаёт случайное подмножество датасета (пары image + label по stem).
    """
    source = Path(source_dir)
    output = Path(output_dir)

    random.seed(seed)

    images_dir = source / "images"
    labels_dir = source / "labels"

    images = sorted(images_dir.glob("*.png"))
    mask_by_stem = {p.stem: p for p in labels_dir.glob("*.png")}

    pairs: list[tuple[Path, Path]] = []
    for img in images:
        mask = mask_by_stem.get(img.stem)
        if mask is not None:
            pairs.append((img, mask))

    print(f"Всего патчей: {len(pairs)}")

    if num_samples > len(pairs):
        num_samples = len(pairs)

    selected = random.sample(pairs, num_samples)

    out_images = output / "images"
    out_labels = output / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print(f"Копирование {num_samples} патчей...")
    for img_src, mask_src in selected:
        shutil.copy(img_src, out_images / img_src.name)
        shutil.copy(mask_src, out_labels / mask_src.name)

    print(f"✅ Создано подмножество: {output}")
    print(f"   Патчей: {num_samples}")
    return num_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/inria_patches")
    parser.add_argument("--output", default="data/inria_subset_1k")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    create_subset(
        PROJECT_ROOT / args.source,
        PROJECT_ROOT / args.output,
        args.num_samples,
        args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
