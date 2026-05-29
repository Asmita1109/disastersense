"""
DisasterSense | Exploratory Data Analysis
Dataset: CrisisMMD v2.0 (Alam et al., 2018)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

sns.set_theme(style="whitegrid", palette="husl")

RAW_DIR    = Path("data/raw/crisismmd_datasplit_all")
IMAGE_BASE = Path("data/raw/CrisisMMD_v2.0")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["task_damage", "task_humanitarian", "task_informative"]


def load_task(task):
    splits = {}
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(RAW_DIR / f"{task}_text_img_{split}.tsv", sep="\t")
        df.columns = df.columns.str.strip()
        splits[split] = df
    return splits


def load_all():
    data = {}
    for task in TASKS:
        data[task] = load_task(task)
        total = sum(len(v) for v in data[task].values())
        print(f"{task:30s} → {total:,} samples")
    return data


def summarise(data):
    print("\n── Summary ───────────────────────────────────────────")
    for task, splits in data.items():
        print(f"\n{task.upper()}")
        for split, df in splits.items():
            print(f"  {split:6s}: {len(df):,}")
        counts = splits["train"]["label"].value_counts()
        ratio  = counts.max() / counts.min()
        print(f"  Labels:\n{counts.to_string()}")
        print(f"  Imbalance ratio: {ratio:.2f}x")
        if ratio > 3:
            print("  → Weighted loss recommended.")
    print("──────────────────────────────────────────────────────\n")


def encode_labels(df):
    classes = sorted(df["label"].unique())
    return {
        "label2idx": {c: i for i, c in enumerate(classes)},
        "idx2label": {i: c for i, c in enumerate(classes)},
    }


def plot_distributions(data):
    fig, axes = plt.subplots(len(TASKS), 3, figsize=(18, 4 * len(TASKS)))
    fig.suptitle("CrisisMMD v2.0 — Label Distributions", fontsize=14, fontweight="bold")
    for row, task in enumerate(TASKS):
        for col, split in enumerate(["train", "dev", "test"]):
            counts = data[task][split]["label"].value_counts()
            axes[row, col].bar(counts.index, counts.values)
            axes[row, col].set_title(f"{task} / {split}")
            axes[row, col].tick_params(axis="x", rotation=20)
            axes[row, col].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distributions.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_events(data):
    counts = data["task_damage"]["train"]["event_name"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts.values,
        labels=[e.replace("_", "\n") for e in counts.index],
        autopct="%1.1f%%",
        colors=sns.color_palette("Set2", len(counts)),
    )
    ax.set_title("Training Samples per Disaster Event", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "events.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_samples(data, n=3):
    train  = data["task_damage"]["train"]
    labels = sorted(train["label"].unique())
    fig, axes = plt.subplots(len(labels), n, figsize=(n * 4, len(labels) * 4))
    fig.suptitle("Sample Images per Damage Class", fontsize=13, fontweight="bold")
    for row, label in enumerate(labels):
        samples = train[train["label"] == label].head(n)
        for col, (_, r) in enumerate(samples.iterrows()):
            ax = axes[row, col]
            try:
                ax.imshow(Image.open(IMAGE_BASE / r["image"]).convert("RGB"))
                ax.set_title(label.replace("_", "\n"), fontsize=8)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "not found", ha="center", va="center")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "samples.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_tweet_lengths(data):
    train = data["task_damage"]["train"].copy()
    train["length"] = train["tweet_text"].astype(str).apply(len)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in sorted(train["label"].unique()):
        ax.hist(train[train["label"] == label]["length"], bins=40, alpha=0.6, label=label)
    ax.set_title("Tweet Length by Damage Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tweet_lengths.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    data = load_all()
    summarise(data)

    label_maps = {task: encode_labels(data[task]["train"]) for task in TASKS}
    for task, m in label_maps.items():
        print(f"{task} → {m['label2idx']}")

    plot_distributions(data)
    plot_events(data)
    plot_samples(data)
    plot_tweet_lengths(data)

    os.makedirs("data/processed", exist_ok=True)
    for split in ["train", "dev", "test"]:
        data["task_damage"][split].to_csv(f"data/processed/damage_{split}.csv", index=False)
    print("Saved → data/processed/")
