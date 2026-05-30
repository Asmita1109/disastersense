"""
DisasterSense | NLP Model Training
Fine-tunes twitter-roberta-base on CrisisMMD informative classification task.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from nlp_preprocess import build_nlp_dataloaders

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
NUM_LABELS = 2
EPOCHS     = 5
BATCH_SIZE = 32
LR         = 2e-5
MODEL_DIR  = Path("models/nlp_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP     = {0: "not_informative", 1: "informative"}

print(f"Device: {DEVICE}")


def build_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
    return model.to(DEVICE)


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        correct    += (outputs.logits.argmax(1) == labels).sum().item()
        total      += input_ids.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs     = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * input_ids.size(0)
            correct    += (outputs.logits.argmax(1) == labels).sum().item()
            total      += input_ids.size(0)

    return total_loss / total, correct / total


def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("NLP Model Training Curves", fontsize=13, fontweight="bold")
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
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            preds.extend(model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(1).cpu().tolist())
            targets.extend(batch["label"].tolist())

    pred_names = [LABEL_MAP[p] for p in preds]
    true_names = [LABEL_MAP[t] for t in targets]

    print("\n── Classification Report ─────────────────────────────")
    print(classification_report(true_names, pred_names))

    cm = confusion_matrix(true_names, pred_names, labels=list(LABEL_MAP.values()))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(LABEL_MAP.values()),
                yticklabels=list(LABEL_MAP.values()), ax=ax)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    loaders, tokenizer = build_nlp_dataloaders(BATCH_SIZE)
    model              = build_model()
    optimizer          = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler          = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0

    print("\n── Training ──────────────────────────────────────────")
    for epoch in range(1, EPOCHS + 1):
        tl, ta = train_epoch(model, loaders["train"], optimizer)
        vl, va = evaluate(model, loaders["dev"])
        scheduler.step()

        for k, v in zip(["train_loss", "val_loss", "train_acc", "val_acc"], [tl, vl, ta, va]):
            history[k].append(v)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Acc: {va:.4f}")

        if va > best_val:
            best_val = va
            model.save_pretrained(MODEL_DIR / "best")
            tokenizer.save_pretrained(MODEL_DIR / "best")
            print(f"  → Saved (val_acc: {best_val:.4f})")

    with open(MODEL_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_curves(history)

    from transformers import AutoModelForSequenceClassification
    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "best").to(DEVICE)
    evaluate_test(best_model, loaders["test"])
    print(f"\nBest val accuracy: {best_val:.4f}")
