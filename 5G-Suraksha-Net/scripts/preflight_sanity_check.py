#!/usr/bin/env python
"""Pre-flight sanity check for Phase 2B VideoMAE training.

Verifies all 8 mandatory preflight conditions:
1. Run one verified training batch.
2. Run one validation batch.
3. Confirm the model output shape (B, 2).
4. Confirm binary labels are {0, 1}.
5. Confirm loss is finite.
6. Confirm gradients are finite.
7. Confirm checkpoint save and load round-trip.
8. Confirm validation metrics calculation is finite.
Also asserts dataset lock numbers and zero-leakage partitions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "outputs" / "phase2b" / "experiment_001"

# Import Dataset from training script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_videomae_fight import VideoMAEClassificationDataset


def run_sanity_check():
    print("=================================================================")
    print("PHASE 2B PRE-FLIGHT SANITY CHECK")
    print("=================================================================")

    # 1. Dataset Lock Verification
    train_df = pd.read_csv(EXP_DIR / "train_manifest.csv")
    val_df = pd.read_csv(EXP_DIR / "val_manifest.csv")
    test_df = pd.read_csv(EXP_DIR / "test_manifest.csv")

    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    total_clips = n_train + n_val + n_test
    total_fight = int((train_df["label"] == 1).sum() + (val_df["label"] == 1).sum() + (test_df["label"] == 1).sum())
    total_normal = int((train_df["label"] == 0).sum() + (val_df["label"] == 0).sum() + (test_df["label"] == 0).sum())

    print(f"Asserting dataset lock numbers:")
    print(f"  Total Clips: {total_clips} (Expected: 7006)")
    print(f"  FIGHT:       {total_fight} (Expected: 2844)")
    print(f"  NORMAL:      {total_normal} (Expected: 4162)")
    print(f"  Train:       {n_train} (Expected: 5041)")
    print(f"  Validation:  {n_val} (Expected: 979)")
    print(f"  Test:        {n_test} (Expected: 986)")

    assert total_clips == 7006, f"Dataset lock mismatch: total_clips = {total_clips}"
    assert total_fight == 2844, f"Dataset lock mismatch: total_fight = {total_fight}"
    assert total_normal == 4162, f"Dataset lock mismatch: total_normal = {total_normal}"
    assert n_train == 5041, f"Dataset lock mismatch: n_train = {n_train}"
    assert n_val == 979, f"Dataset lock mismatch: n_val = {n_val}"
    assert n_test == 986, f"Dataset lock mismatch: n_test = {n_test}"
    print("  --> [OK] Dataset lock assertion passed.")

    # 2. Leakage Assertions
    tr_groups = set(train_df["group_id"])
    va_groups = set(val_df["group_id"])
    te_groups = set(test_df["group_id"])

    assert len(tr_groups & va_groups) == 0, "Leakage: Train/Val group overlap!"
    assert len(tr_groups & te_groups) == 0, "Leakage: Train/Test group overlap!"
    assert len(va_groups & te_groups) == 0, "Leakage: Val/Test group overlap!"
    print("  --> [OK] Group isolation assertion passed (zero movie/source leakage).")

    # 3. Model & Hardware Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "CUDA device not available!"
    print(f"  --> [OK] Device: {device} ({torch.cuda.get_device_name(0)})")

    pretrained_path = PROJECT_ROOT / "models" / "pretrained_videomae_small"
    model = VideoMAEForVideoClassification.from_pretrained(
        str(pretrained_path),
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)

    # 4. Checkpoint 1 & 2: Verified Training Batch & Validation Batch
    train_ds = VideoMAEClassificationDataset(EXP_DIR / "train_manifest.csv", is_train=True)
    val_ds = VideoMAEClassificationDataset(EXP_DIR / "val_manifest.csv", is_train=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    train_bx, train_by = next(iter(train_loader))
    val_bx, val_by = next(iter(val_loader))

    # 5. Check 3: Output shape (B, 2)
    model.train()
    train_bx = train_bx.to(device)
    train_by = train_by.to(device)

    with autocast("cuda"):
        out = model(pixel_values=train_bx, labels=train_by)
        logits = out.logits
        loss = out.loss

    assert logits.shape == (4, 2), f"Unexpected output shape: {logits.shape}"
    print(f"  --> [OK] Model output shape verified: {tuple(logits.shape)}")

    # 6. Check 4: Binary labels are {0, 1}
    unique_labels = set(train_by.cpu().numpy().tolist())
    assert unique_labels.issubset({0, 1}), f"Non-binary labels found: {unique_labels}"
    print(f"  --> [OK] Binary labels verified: {unique_labels}")

    # 7. Check 5: Loss is finite
    assert torch.isfinite(loss).item(), f"Non-finite loss: {loss}"
    print(f"  --> [OK] Loss is finite: {loss.item():.4f}")

    # 8. Check 6: Gradients are finite
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all().item(), f"Non-finite gradient in {name}"
    print("  --> [OK] Gradients are finite across all parameters.")

    # 9. Check 7: Checkpoint save and load
    test_ckpt_path = EXP_DIR / "sanity_check_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, test_ckpt_path)
    loaded_ckpt = torch.load(test_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(loaded_ckpt["model_state_dict"])
    if test_ckpt_path.exists():
        test_ckpt_path.unlink()
    print("  --> [OK] Checkpoint save/load round-trip verified.")

    # 10. Check 8: Validation batch & metrics finite
    model.eval()
    val_bx = val_bx.to(device)
    val_by = val_by.to(device)
    with torch.no_grad(), autocast("cuda"):
        val_out = model(pixel_values=val_bx)
        val_logits = val_out.logits
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()

    val_preds = (val_probs >= 0.5).astype(int)
    targets = val_by.cpu().numpy()

    p = precision_score(targets, val_preds, zero_division=0)
    r = recall_score(targets, val_preds, zero_division=0)
    f = f1_score(targets, val_preds, zero_division=0)
    pr_auc = average_precision_score(targets, val_probs) if len(np.unique(targets)) > 1 else 0.5
    roc_auc = roc_auc_score(targets, val_probs) if len(np.unique(targets)) > 1 else 0.5

    assert np.isfinite(p) and np.isfinite(r) and np.isfinite(f) and np.isfinite(pr_auc) and np.isfinite(roc_auc)
    print(f"  --> [OK] Validation metrics are finite: P={p:.2f}, R={r:.2f}, F1={f:.2f}, PR-AUC={pr_auc:.2f}")

    print("=================================================================")
    print("ALL 8 PRE-FLIGHT SANITY CHECKS PASSED. READY FOR FULL 20-EPOCH RUN.")
    print("=================================================================")
    return True


if __name__ == "__main__":
    success = run_sanity_check()
    sys.exit(0 if success else 1)
