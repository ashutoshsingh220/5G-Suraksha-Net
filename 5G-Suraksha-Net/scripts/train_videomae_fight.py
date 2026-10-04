#!/usr/bin/env python
"""Phase 2B — Video Fight Model Training & Epoch-wise Validation Engine.

Architecture: VideoMAE-Small (ViT-S, 21.9M parameters, 16 frames @ 224x224, stride 2).
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM), FP16 mixed precision.
Schedule:
  - max_epochs = 20
  - min_epochs = 8
  - early_stopping = enabled (patience = 5 after min_epochs)
  - learning_rate scheduler: ReduceLROnPlateau monitoring val_pr_auc (mode='max', factor=0.5, patience=2)
  - checkpoints: best_model.pt (selected on val_pr_auc), latest_model.pt (every epoch)
  - full logging: every epoch records train_loss, val_loss, val_precision, val_recall,
                  val_f1, val_roc_auc, val_pr_auc, val_fpr, val_fnr, lr, time, vram.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "outputs" / "phase2b" / "experiment_001"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Kinetics / ImageNet normalization constants
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VideoMAEClassificationDataset(Dataset):
    """Dataset for video clips, reading 16 frames with stride 2 and robust error handling."""

    def __init__(self, manifest_csv: str | Path, num_frames: int = 16, stride: int = 2, is_train: bool = False):
        self.df = pd.read_csv(manifest_csv)
        self.num_frames = num_frames
        self.stride = stride
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _read_video_frames(self, path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        span = self.num_frames * self.stride

        # Determine start frame
        if total_frames > span:
            if self.is_train:
                start_frame = np.random.randint(0, total_frames - span)
            else:
                start_frame = (total_frames - span) // 2
        else:
            start_frame = 0

        # Target frame indices
        if total_frames >= span:
            target_indices = set(start_frame + i * self.stride for i in range(self.num_frames))
            max_idx = start_frame + (self.num_frames - 1) * self.stride
        else:
            if total_frames > 0:
                target_indices = set(np.linspace(0, total_frames - 1, self.num_frames).astype(int).tolist())
            else:
                target_indices = {0}
            max_idx = total_frames - 1

        frames = []
        cur_frame_idx = 0
        while cur_frame_idx <= max_idx and len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if cur_frame_idx in target_indices:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cur_frame_idx += 1
        cap.release()

        # If underrun, pad with last frame or black
        while len(frames) < self.num_frames:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        frames_arr = np.array(frames[:self.num_frames], dtype=np.float32) / 255.0

        # Augmentation for training only:
        if self.is_train:
            # 1. Random horizontal flip (p=0.5)
            if random.random() < 0.5:
                frames_arr = np.flip(frames_arr, axis=2).copy()
            # 2. Subtle brightness jitter (p=0.3, factor 0.85 - 1.15)
            if random.random() < 0.3:
                bright_factor = random.uniform(0.85, 1.15)
                frames_arr = np.clip(frames_arr * bright_factor, 0.0, 1.0)

        # Normalize
        frames_arr = (frames_arr - NORM_MEAN) / NORM_STD
        # Transpose: (16, 224, 224, 3) -> (16, 3, 224, 224)
        tensor = torch.from_numpy(frames_arr).permute(0, 3, 1, 2)
        return tensor

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = str(row["video_path"])
        label = int(row["label"])
        tensor = self._read_video_frames(path)
        return tensor, label


def evaluate(model, data_loader, device, criterion):
    """Run evaluation and compute full forensic metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with autocast("cuda"):
                outputs = model(pixel_values=batch_x)
                logits = outputs.logits
                loss = criterion(logits, batch_y)

            total_loss += loss.item() * len(batch_y)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / max(len(all_targets), 1)
    targets = np.array(all_targets)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    precision = float(precision_score(targets, preds, zero_division=0))
    recall = float(recall_score(targets, preds, zero_division=0))
    f1 = float(f1_score(targets, preds, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(targets, probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(targets, probs))
    except Exception:
        pr_auc = float(np.mean(targets))

    cm = confusion_matrix(targets, preds, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = float(fp / max(fp + tn, 1))
    fnr = float(fn / max(fn + tp, 1))

    return {
        "loss": round(avg_loss, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def train_model(
    max_epochs: int = 20,
    min_epochs: int = 8,
    patience: int = 5,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-5,
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Assert Dataset Lock
    train_csv = EXP_DIR / "train_manifest.csv"
    val_csv = EXP_DIR / "val_manifest.csv"
    test_csv = EXP_DIR / "test_manifest.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    assert len(train_df) == 5041, f"Train count mismatch: {len(train_df)} != 5041"
    assert len(val_df) == 979, f"Val count mismatch: {len(val_df)} != 979"
    assert len(test_df) == 986, f"Test count mismatch: {len(test_df)} != 986"

    # Assert zero leakage
    tr_groups = set(train_df["group_id"])
    va_groups = set(val_df["group_id"])
    te_groups = set(test_df["group_id"])
    assert len(tr_groups & va_groups) == 0, "Leakage: Train/Val overlap"
    assert len(tr_groups & te_groups) == 0, "Leakage: Train/Test overlap"
    assert len(va_groups & te_groups) == 0, "Leakage: Val/Test overlap"
    print("Dataset lock and zero-leakage verified.")

    # Save config.yaml
    config_dict = {
        "experiment_name": "phase2b_videomae_small_001",
        "model_architecture": "VideoMAE-Small (ViT-S)",
        "num_frames": 16,
        "frame_size": [224, 224],
        "temporal_stride": 2,
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "initial_learning_rate": learning_rate,
        "lr_scheduler": "ReduceLROnPlateau(mode='max', factor=0.5, patience=2, min_lr=1e-6)",
        "optimizer": "AdamW(weight_decay=0.05)",
        "mixed_precision": "FP16 (torch.amp.autocast)",
        "dataset_counts": {
            "train": 5041,
            "val": 979,
            "test": 986,
            "total": 7006,
        },
    }
    with open(EXP_DIR / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    train_ds = VideoMAEClassificationDataset(train_csv, is_train=True)
    val_ds = VideoMAEClassificationDataset(val_csv, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load pretrained VideoMAE-Small
    pretrained_path = PROJECT_ROOT / "models" / "pretrained_videomae_small"
    print(f"Loading pretrained VideoMAE-Small from {pretrained_path}...")
    model = VideoMAEForVideoClassification.from_pretrained(
        str(pretrained_path),
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    # ReduceLROnPlateau monitoring val_pr_auc
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = GradScaler("cuda")

    history = []
    best_pr_auc = -1.0
    best_epoch = 0
    patience_counter = 0

    history_csv = EXP_DIR / "training_history.csv"
    history_json = EXP_DIR / "training_history.json"
    best_model_path = EXP_DIR / "best_model.pt"
    latest_model_path = EXP_DIR / "latest_model.pt"

    print("=================================================================")
    print(f"STARTING PHASE 2B VIDEOMAE TRAINING (MAX {max_epochs} EPOCHS, MIN {min_epochs} EPOCHS)")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Batch size: {batch_size}, Grad accum: {grad_accum_steps} (Effective: {batch_size * grad_accum_steps})")
    print(f"Early Stopping: patience={patience} after epoch {min_epochs} monitoring val_pr_auc")
    print("=================================================================")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_count = 0
        optimizer.zero_grad()

        for step, (bx, by) in enumerate(train_loader):
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            with autocast("cuda"):
                outputs = model(pixel_values=bx, labels=by)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * grad_accum_steps * len(by)
            train_count += len(by)

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_train_loss = round(train_loss / max(train_count, 1), 4)

        # Validation
        val_metrics = evaluate(model, val_loader, device, criterion)
        epoch_time = round(time.time() - t0, 1)
        peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        # Step scheduler based on val_pr_auc
        scheduler.step(val_metrics["pr_auc"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_fpr": val_metrics["fpr"],
            "val_fnr": val_metrics["fnr"],
            "learning_rate": round(current_lr, 7),
            "epoch_duration": epoch_time,
            "peak_vram": peak_vram,
        }
        history.append(epoch_record)

        # Display full metrics line
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} [{epoch_time}s] | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.4f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} | "
            f"val_fpr={val_metrics['fpr']:.4f} | "
            f"val_fnr={val_metrics['fnr']:.4f} | "
            f"lr={current_lr:.7f} | "
            f"peak_vram={peak_vram} MB"
        )

        # Save latest checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "config": model.config.to_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            latest_model_path,
        )

        # Checkpoint selection strictly on validation PR-AUC
        if val_metrics["pr_auc"] > best_pr_auc:
            improvement = val_metrics["pr_auc"] - best_pr_auc
            best_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": model.config.to_dict(),
                    "best_val_pr_auc": best_pr_auc,
                },
                best_model_path,
            )
            print(f"  --> [*] New best model saved at epoch {epoch} (val_pr_auc: {best_pr_auc:.4f}, +{improvement:.4f})")
        else:
            patience_counter += 1
            print(f"  --> [i] No improvement in val_pr_auc (best: {best_pr_auc:.4f} at epoch {best_epoch}). Patience: {patience_counter}/{patience}")

        # Save history after every epoch
        with open(history_json, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        with open(history_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        # Early stopping check: only after min_epochs
        if epoch >= min_epochs and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}: No val_pr_auc improvement for {patience} consecutive epochs after min_epochs ({min_epochs}).")
            break

    print("=================================================================")
    print(f"TRAINING COMPLETE.")
    print(f"Total Epochs Completed: {len(history)}")
    print(f"Best Epoch:             {best_epoch}")
    print(f"Best Validation PR-AUC: {best_pr_auc:.4f}")
    print(f"Best Checkpoint:        {best_model_path}")
    print(f"Latest Checkpoint:      {latest_model_path}")
    print("=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    train_model(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
    )
