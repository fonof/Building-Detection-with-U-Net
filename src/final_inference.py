"""
Финальный инференс для портфолио.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import BuildingDataset, get_normalize_stats, get_transforms
from src.model import create_model
from src.utils import ensure_dir, log_info, log_ok


def _denormalize_image(tensor: torch.Tensor, dataset_name: str) -> np.ndarray:
    """Tensor (3,H,W) -> float RGB [0, 1] для визуализации."""
    mean, std = get_normalize_stats(dataset_name)
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * std[c] + mean[c]
    return np.clip(img, 0, 1)


def get_val_paths(
    images_dir: Path,
    masks_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Val split — тот же seed/ratio, что в create_dataloaders."""
    from src.dataset import _collect_pairs

    all_images, all_masks = _collect_pairs(images_dir, masks_dir)
    n = len(all_images)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    train_size = max(1, int(n * train_ratio))
    if n - train_size == 0:
        train_size = n - 1
    val_indices = perm[train_size:]
    return (
        [all_images[i] for i in val_indices],
        [all_masks[i] for i in val_indices],
    )


def run_inference(
    model_path: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path,
    output_dir: str | Path,
    num_samples: int = 10,
    threshold: float = 0.5,
    dataset_name: str = "inria",
    use_val_split: bool = True,
    seed: int = 42,
) -> dict[str, float]:
    """Запускает инференс и создаёт визуализации."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device=device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if use_val_split:
        val_images, val_masks = get_val_paths(images_dir, masks_dir)
        dataset = BuildingDataset(
            images_dir,
            masks_dir,
            img_size=256,
            transform=get_transforms(False, 256, dataset_name),
            image_paths=val_images,
            mask_paths=val_masks,
        )
        log_info(f"Val split: {len(dataset)} пар (held-out 20%)")
    else:
        dataset = BuildingDataset(
            images_dir,
            masks_dir,
            img_size=256,
            transform=get_transforms(False, 256, dataset_name),
        )

    output_path = ensure_dir(output_dir, create=True)
    n_samples = min(num_samples, len(dataset))
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n_samples)

    fig, axes = plt.subplots(n_samples, 5, figsize=(25, 5 * n_samples))
    if n_samples == 1:
        axes = np.array([axes])

    metrics: list[dict[str, float]] = []

    with torch.inference_mode():
        for row, sample_idx in enumerate(indices):
            image, mask = dataset[sample_idx]

            image_tensor = image.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(image_tensor)).squeeze().cpu().numpy()
            pred_binary = (pred > threshold).astype(np.float32)

            img_np = _denormalize_image(image, dataset_name)
            mask_np = mask.squeeze().numpy()

            intersection = (pred_binary * mask_np).sum()
            union = pred_binary.sum() + mask_np.sum() - intersection
            iou = intersection / (union + 1e-6)

            precision = intersection / (pred_binary.sum() + 1e-6)
            recall = intersection / (mask_np.sum() + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            metrics.append(
                {"iou": iou, "precision": precision, "recall": recall, "f1": f1}
            )

            overlay = (img_np * 255).astype(np.uint8).copy()
            overlay[mask_np > 0.5] = [0, 255, 0]
            overlay[(pred_binary > 0.5) & (mask_np < 0.5)] = [255, 0, 0]
            overlay[(pred_binary < 0.5) & (mask_np > 0.5)] = [0, 0, 255]

            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title(f"Image {row + 1}", fontsize=10)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_np, cmap="gray")
            axes[row, 1].set_title(
                f"Ground Truth\n({mask_np.mean() * 100:.1f}%)", fontsize=10
            )
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred, cmap="hot", vmin=0, vmax=1)
            axes[row, 2].set_title(
                f"Prediction (raw)\n({pred.mean():.3f})", fontsize=10
            )
            axes[row, 2].axis("off")

            axes[row, 3].imshow(pred_binary, cmap="gray")
            axes[row, 3].set_title(
                f"Prediction (binary)\n({pred_binary.mean() * 100:.1f}%)",
                fontsize=10,
            )
            axes[row, 3].axis("off")

            axes[row, 4].imshow(overlay)
            axes[row, 4].set_title(
                f"Overlay\nIoU: {iou:.3f}\nF1: {f1:.3f}", fontsize=10
            )
            axes[row, 4].axis("off")

            cv2.imwrite(
                str(output_path / f"sample_{row + 1:02d}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

    plt.tight_layout()
    grid_path = output_path / "all_predictions.png"
    fig.savefig(str(grid_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_ok(f"Сохранено: {grid_path}")

    print("\n📊 МЕТРИКИ НА ТЕСТОВЫХ ОБРАЗЦАХ:")
    print("=" * 60)
    for idx, m in enumerate(metrics):
        print(
            f"Sample {idx + 1}: IoU={m['iou']:.4f}, Precision={m['precision']:.4f}, "
            f"Recall={m['recall']:.4f}, F1={m['f1']:.4f}"
        )

    avg_iou = float(np.mean([m["iou"] for m in metrics]))
    avg_f1 = float(np.mean([m["f1"] for m in metrics]))
    avg_precision = float(np.mean([m["precision"] for m in metrics]))
    avg_recall = float(np.mean([m["recall"] for m in metrics]))

    print("\n" + "=" * 60)
    print(f"СРЕДНИЕ: IoU={avg_iou:.4f}, F1={avg_f1:.4f}, "
          f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}")
    print("=" * 60)

    return {
        "avg_iou": avg_iou,
        "avg_f1": avg_f1,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Финальный инференс для портфолио")
    parser.add_argument(
        "--model_path",
        default="models/inria_10k_best.pth",
        help="Путь к весам модели",
    )
    parser.add_argument(
        "--images_dir",
        default="data/inria_subset_10k/images",
        help="Папка с изображениями",
    )
    parser.add_argument(
        "--masks_dir",
        default="data/inria_subset_10k/labels",
        help="Папка с масками (GT)",
    )
    parser.add_argument(
        "--output_dir",
        default="output/portfolio",
        help="Папка для результатов",
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dataset_name", default="inria")
    parser.add_argument(
        "--use_val_split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Использовать val split 20%% (как при обучении)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_inference(
        model_path=PROJECT_ROOT / args.model_path,
        images_dir=PROJECT_ROOT / args.images_dir,
        masks_dir=PROJECT_ROOT / args.masks_dir,
        output_dir=PROJECT_ROOT / args.output_dir,
        num_samples=args.num_samples,
        threshold=args.threshold,
        dataset_name=args.dataset_name,
        use_val_split=args.use_val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
