"""
DisasterSense | Exploratory Data Analysis
==========================================
Dataset : CrisisMMD v2.0 (Alam et al., 2018)
Tasks   : Damage Severity | Humanitarian | Informative
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

sns.set_theme(style="whitegrid", palette="husl")

# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_DIR    = Path("data/raw/crisismmd_datasplit_all/crisismmd_datasplit_all")
IMAGE_BASE = Path("data/raw/CrisisMMD_v2.0")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["task_damage", "task_humanitarian", "task_informative"]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_task(task: str) -> dict:
    """Load train/dev/test splits for a given task."""
    splits = {}
    for split in ["train", "dev", "test"]:
        path = RAW_DIR / f"{task}_text_img_{split}.tsv"
        df = pd.read_csv(path, sep="\t")
        df.columns = df.columns.str.strip()
        splits[split] = df
    return splits


def load_all_tasks() -> dict:
    """Load all three tasks and their splits."""
    data = {}
    for task in TASKS:
        data[task] = load_task(task)
        total = sum(len(v) for v in data[task].values())
        print(f"Loaded {task:30s} → {total:,} samples")
    return data


# ── Analysis ──────────────────────────────────────────────────────────────────

def summarise(data: dict) -> None:
    """Print split sizes, label distributions and imbalance ratios per task."""
    print("\n── Dataset Summary ───────────────────────────────────────")
    for task, splits in data.items():
        print(f"\n{task.upper()}")
        for split, df in splits.items():
            print(f"  {split:6s}: {len(df):,} samples")
        train  = splits["train"]
        counts = train["label"].value_counts()
        ratio  = counts.max() / counts.min()
        print(f"  Label distribution (train):\n{counts.to_string()}")
        print(f"  Imbalance ratio: {ratio:.2f}x")
        if ratio > 3:
            print("  → Weighted loss recommended.")
    print("──────────────────────────────────────────────────────────\n")


def encode_labels(df: pd.DataFrame) -> dict:
    """Return label-to-index and index-to-label mappings."""
    classes = sorted(df["label"].unique())
    return {
        "label2idx": {c: i for i, c in enumerate(classes)},
        "idx2label": {i: c for i, c in enumerate(classes)},
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_label_distributions(data: dict) -> None:
    """Bar charts of label distributions across all tasks and splits."""
    fig, axes = plt.subplots(len(TASKS), 3, figsize=(18, 4 * len(TASKS)))
    fig.suptitle("CrisisMMD v2.0 — Label Distributions", fontsize=14, fontweight="bold")

    for row, task in enumerate(TASKS):
        for col, split in enumerate(["train", "dev", "test"]):
            df     = data[task][split]
            counts = df["label"].value_counts()
            axes[row, col].bar(counts.index, counts.values)
            axes[row, col].set_title(f"{task} / {split}")
            axes[row, col].tick_params(axis="x", rotation=20)
            axes[row, col].set_ylabel("Count")

    plt.tight_layout()
    path = OUTPUT_DIR / "label_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


def plot_event_distribution(data: dict) -> None:
    """Pie chart showing sample count per disaster event."""
    train  = data["task_damage"]["train"]
    counts = train["event_name"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts.values,
        labels=[e.replace("_", "\n") for e in counts.index],
        autopct="%1.1f%%",
        colors=sns.color_palette("Set2", len(counts)),
    )
    ax.set_title("Training Samples per Disaster Event", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = OUTPUT_DIR / "event_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


def plot_sample_images(data: dict, n_per_class: int = 3) -> None:
    """Display sample images per damage label from the training set."""
    train  = data["task_damage"]["train"]
    labels = sorted(train["label"].unique())

    fig, axes = plt.subplots(
        len(labels), n_per_class,
        figsize=(n_per_class * 4, len(labels) * 4)
    )
    fig.suptitle("Sample Images per Damage Class", fontsize=13, fontweight="bold")

    for row, label in enumerate(labels):
        samples = train[train["label"] == label].head(n_per_class)
        for col, (_, row_data) in enumerate(samples.iterrows()):
            img_path = IMAGE_BASE / row_data["image"]
            ax = axes[row, col]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(label.replace("_", "\n"), fontsize=8)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()
    path = OUTPUT_DIR / "sample_images.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


def plot_tweet_length_distribution(data: dict) -> None:
    """Histogram of tweet text lengths across damage severity classes."""
    train = data["task_damage"]["train"].copy()
    train["text_length"] = train["tweet_text"].astype(str).apply(len)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label in sorted(train["label"].unique()):
        subset = train[train["label"] == label]["text_length"]
        ax.hist(subset, bins=40, alpha=0.6, label=label)

    ax.set_title("Tweet Length Distribution by Damage Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tweet Length (characters)")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    path = OUTPUT_DIR / "tweet_lengths.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_all_tasks()

    summarise(data)

    label_maps = {task: encode_labels(data[task]["train"]) for task in TASKS}
    for task, maps in label_maps.items():
        print(f"{task} → {maps['label2idx']}")

    plot_label_distributions(data)
    plot_event_distribution(data)
    plot_sample_images(data)
    plot_tweet_length_distribution(data)

    os.makedirs("data/processed", exist_ok=True)
    data["task_damage"]["train"].to_csv("data/processed/damage_train.csv", index=False)
    data["task_damage"]["dev"].to_csv("data/processed/damage_dev.csv", index=False)
    data["task_damage"]["test"].to_csv("data/processed/damage_test.csv", index=False)
    print("\nSaved processed splits → data/processed/")
