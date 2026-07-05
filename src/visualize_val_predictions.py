"""
Визуализация предсказаний на валидации.
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

from src.dataset import create_dataloaders, get_normalize_stats
from src.model import create_model
from src.utils import ensure_dir, log_ok


def denormalize(tensor: torch.Tensor, dataset_name: str = "inria") -> np.ndarray:
    """Tensor (3,H,W) normalized -> uint8 RGB."""
    mean, std = get_normalize_stats(dataset_name)
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * std[c] + mean[c]
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def visualize_predictions(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    num_samples: int = 5,
    save_dir: str | Path = "output/val_preds",
    dataset_name: str = "inria",
) -> None:
    save_path = Path(save_dir)
    ensure_dir(save_path, create=True)

    model.eval()
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs)

        n = min(num_samples, images.size(0))
        fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
        if n == 1:
            axes = np.array([axes])

        for i in range(n):
            img = denormalize(images[i], dataset_name)
            mask = masks[i].squeeze().cpu().numpy()
            pred = preds[i].squeeze().cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8)

            overlay = img.copy()
            overlay[mask > 0.5] = [0, 255, 0]
            overlay[pred_binary > 0] = [255, 0, 0]

            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Image {i}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask, cmap="gray")
            axes[i, 1].set_title(f"Ground Truth\n({mask.mean() * 100:.1f}%)")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred, cmap="hot", vmin=0, vmax=1)
            axes[i, 2].set_title(f"Prediction (raw)\n({pred.mean():.3f})")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title("Overlay\n(GT=green, Pred=red)")
            axes[i, 3].axis("off")

            cv2.imwrite(
                str(save_path / f"sample_{i}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

        plt.tight_layout()
        fig.savefig(save_path / "all_samples.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log_ok(f"Сохранено в {save_path}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = PROJECT_ROOT / "models/inria_10k_best.pth"

    model = create_model(device=device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True),
        strict=False,
    )

    _, val_loader = create_dataloaders(
        PROJECT_ROOT / "data/inria_subset_10k/images",
        PROJECT_ROOT / "data/inria_subset_10k/labels",
        batch_size=4,
        train_ratio=0.8,
        img_size=256,
        dataset_name="inria",
    )

    visualize_predictions(
        model,
        val_loader,
        device,
        save_dir=PROJECT_ROOT / "output/val_preds",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
