"""
Проверка IoU при разных порогах бинаризации.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_dataloaders
from src.model import create_model


@torch.no_grad()
def test_thresholds(
    model_path: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path,
    device: torch.device | None = None,
    thresholds: list[float] | None = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6]

    model = create_model(device=device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True),
        strict=False,
    )
    model.eval()

    _, val_loader = create_dataloaders(
        images_dir,
        masks_dir,
        batch_size=4,
        img_size=256,
        dataset_name="inria",
    )

    print(f"Model: {model_path}")
    print(f"Val batches: {len(val_loader)}\n")

    for thresh in thresholds:
        total_iou = 0.0
        count = 0

        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            preds = torch.sigmoid(model(images))
            preds_binary = (preds > thresh).float()

            intersection = (preds_binary * masks).sum(dim=(1, 2, 3))
            union = preds_binary.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
            iou = (intersection / (union + 1e-6)).mean()

            total_iou += iou.item()
            count += 1

        print(f"Threshold {thresh:.1f}: Val IoU = {total_iou / count:.4f}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_thresholds(
        PROJECT_ROOT / "models/inria_10k_best.pth",
        PROJECT_ROOT / "data/inria_subset_10k/images",
        PROJECT_ROOT / "data/inria_subset_10k/labels",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
