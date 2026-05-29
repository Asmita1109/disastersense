"""
DisasterSense | Image Model Training
Fine-tunes EfficientNet-B0 on CrisisMMD damage severity classification.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import build_dataloaders, compute_class_weights, LABEL_MAP, PROCESSED

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(LABEL_MAP)
EPOCHS      = 15
BATCH_SIZE  = 32
LR          = 3e-4
MODEL_DIR   = Path("models/image_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")


def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features       = model.classifier[1].in_features
    model.classifier  = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model.to(DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (model(images).argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs        = model(images)
            total_loss    += criterion(outputs, labels).item() * images.size(0)
            correct       += (outputs.argmax(1) == labels).sum().item()
            total         += images.size(0)
    return total_loss / total, correct / total


def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves", fontsize=13, fontweight="bold")
    for ax, metric in zip(axes, ["loss", "acc"]):
        ax.plot(history[f"train_{metric}"], label="Train")
        ax.plot(history[f"val_{metric}"],   label="Val")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "curves.png", dpi=150, bbox_inches="tight")
    plt.show()


def evaluate_test(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds.extend(model(images.to(DEVICE)).argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    idx2label   = {v: k for k, v in LABEL_MAP.items()}
    pred_names  = [idx2label[p] for p in preds]
    true_names  = [idx2label[t] for t in targets]
    print("\n── Classification Report ─────────────────────────────")
    print(classification_report(true_names, pred_names))
    cm = confusion_matrix(true_names, pred_names, labels=sorted(LABEL_MAP.keys()))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(LABEL_MAP.keys()),
                yticklabels=sorted(LABEL_MAP.keys()), ax=ax)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    loaders   = build_dataloaders(BATCH_SIZE)
    weights   = compute_class_weights(PROCESSED / "damage_train.csv").to(DEVICE)
    model     = build_model()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.classifier.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0

    print("\n── Training ──────────────────────────────────────────")
    for epoch in range(1, EPOCHS + 1):
        tl, ta = train_epoch(model, loaders["train"], criterion, optimizer)
        vl, va = evaluate(model, loaders["dev"], criterion)
        scheduler.step()
        for k, v in zip(["train_loss","val_loss","train_acc","val_acc"], [tl,vl,ta,va]):
            history[k].append(v)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Acc: {va:.4f}")
        if va > best_val:
            best_val = va
            torch.save(model.state_dict(), MODEL_DIR / "best.pt")
            print(f"  → Saved (val_acc: {best_val:.4f})")

    with open(MODEL_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_curves(history)

    model.load_state_dict(torch.load(MODEL_DIR / "best.pt"))
    evaluate_test(model, loaders["test"])
    print(f"\nBest val accuracy: {best_val:.4f}")
