"""
Отладка расчёта IoU на валидации.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_dataloaders, get_transforms
from src.model import create_model, get_metrics


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def print_val_transforms() -> None:
    val_tf = get_transforms(is_train=False, img_size=256, dataset_name="inria")
    print("Val transforms:")
    for t in val_tf.transforms:
        print(f"  - {t.__class__.__name__}")


def debug_predictions(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    num_batches: int = 2,
) -> None:
    model.eval()
    smp_iou_fn = get_metrics()[0]

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            print(f"\n--- Batch {batch_idx} ---")
            print(f"Images shape: {images.shape}")
            print(f"Masks shape:  {masks.shape}")
            print(f"Outputs shape: {outputs.shape}")

            print("\nMasks stats:")
            print(f"  Min: {masks.min():.3f}, Max: {masks.max():.3f}, Mean: {masks.mean():.3f}")
            print(f"  Coverage (>0.5): {(masks > 0.5).float().mean():.3f}")

            print("\nPredictions stats:")
            print(f"  Min: {preds.min():.3f}, Max: {preds.max():.3f}, Mean: {preds.mean():.3f}")
            print(f"  Coverage (>0.5): {(preds > 0.5).float().mean():.3f}")

            batch_smp_iou = float(smp_iou_fn(outputs, masks))
            print(f"\nSMP IoU (batch, logits): {batch_smp_iou:.4f}")

            for i in range(preds.shape[0]):
                pred_i = preds[i].squeeze()
                mask_i = masks[i].squeeze()
                iou_manual = calculate_iou(pred_i, mask_i)
                iou_smp = float(smp_iou_fn(outputs[i : i + 1], masks[i : i + 1]))
                print(f"\n  Sample {i}:")
                print(f"    Pred coverage: {(pred_i > 0.5).float().mean():.3f}")
                print(f"    Mask coverage: {(mask_i > 0.5).float().mean():.3f}")
                print(f"    IoU manual:    {iou_manual:.4f}")
                print(f"    IoU SMP:       {iou_smp:.4f}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print_val_transforms()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = PROJECT_ROOT / "models/inria_10k_best.pth"

    model = create_model(device=device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"\n[WARN] Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    _, val_loader = create_dataloaders(
        PROJECT_ROOT / "data/inria_subset_10k/images",
        PROJECT_ROOT / "data/inria_subset_10k/labels",
        batch_size=4,
        train_ratio=0.8,
        img_size=256,
        dataset_name="inria",
    )

    print(f"\nVal batches: {len(val_loader)}, device: {device}")
    debug_predictions(model, val_loader, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
