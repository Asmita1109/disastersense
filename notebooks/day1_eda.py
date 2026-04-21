# =============================================================================
# DisasterSense | Week 1, Day 1 — Dataset Setup & EDA
# =============================================================================
# WHAT WE'RE DOING TODAY:
#   1. Understand the CrisisMMD dataset structure
#   2. Load and inspect images + labels
#   3. Visualize class distribution (are classes balanced?)
#   4. Visualize sample images per class
#   5. Check image sizes (important for preprocessing later)
#
# CONCEPTS COVERED:
#   - Class imbalance and why it matters
#   - What EDA means for image datasets
#   - Label encoding
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import Counter

# =============================================================================
# STEP 1: DATASET — CrisisMMD
# =============================================================================
# CrisisMMD was created by researchers at IIIT Hyderabad.
# It contains tweet images + text from 7 real disaster events:
#   - Hurricane Harvey (2017)
#   - Hurricane Irma (2017)
#   - Hurricane Maria (2017)
#   - Iraq Iran Earthquake (2017)
#   - Mexico Earthquake (2017)
#   - Srilanka Floods (2017)
#   - California Wildfires (2018)
#
# HOW TO DOWNLOAD:
#   1. Go to: https://crisismmD.github.io/
#   2. Fill the form to get the download link (academic use, free)
#   OR use the HuggingFace mirror:
#   >> from datasets import load_dataset
#   >> ds = load_dataset("srajp/CrisisMMD")  # community upload
#
# For now, we'll simulate the structure so you can run this
# immediately and swap in real data when downloaded.
# =============================================================================

# --- Simulate dataset structure for immediate use ---
# When you download CrisisMMD, your folder will look like:
#   data/raw/CrisisMMD/
#       ├── annotations/
#       │   ├── hurricane_harvey_final_data.tsv
#       │   ├── iraq_iran_earthquake_final_data.tsv
#       │   └── ... (one TSV per disaster)
#       └── data_image/
#           ├── hurricane_harvey/
#           │   ├── image_001.jpg
#           │   └── ...
#           └── ...

def load_crisisMMD_annotations(data_dir: str) -> pd.DataFrame:
    """
    Load all annotation TSV files from CrisisMMD into one DataFrame.
    
    CONCEPT — TSV files:
    Tab-Separated Values. Like CSV but uses tab instead of comma.
    CrisisMMD stores labels this way: image path, tweet text, label columns.
    """
    data_dir = Path(data_dir)
    annotation_dir = data_dir / "annotations"
    
    all_dfs = []
    
    # Each disaster has its own annotation file
    for tsv_file in annotation_dir.glob("*.tsv"):
        df = pd.read_csv(tsv_file, sep="\t")
        # Tag which disaster this came from (useful for analysis)
        df["disaster_name"] = tsv_file.stem.replace("_final_data", "")
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Loaded {len(combined)} total samples from {len(all_dfs)} disasters")
    return combined


def simulate_dataset() -> pd.DataFrame:
    """
    Simulates what CrisisMMD looks like so you can run EDA code
    immediately. Replace with load_crisisMMD_annotations() once downloaded.
    
    CONCEPT — Why simulate?
    Always good practice to build and test your pipeline on synthetic data
    first. This way you know your code works before debugging data issues.
    """
    np.random.seed(42)
    n = 1000
    
    disaster_types = [
        "hurricane_harvey", "hurricane_irma", "hurricane_maria",
        "iraq_iran_earthquake", "mexico_earthquake",
        "srilanka_floods", "california_wildfires"
    ]
    
    # CrisisMMD has two key label columns:
    # 1. image_humanitarian — is the image humanitarian-relevant?
    # 2. image_damage — what level of damage is shown?
    humanitarian_labels = [
        "affected_injured_or_dead_people",
        "infrastructure_and_utility_damage",
        "rescue_volunteering_or_donation_effort",
        "vehicle_damage",
        "other_relevant_information",
        "not_humanitarian"
    ]
    
    damage_labels = [
        "severe_damage",
        "mild_damage", 
        "little_or_no_damage"
    ]
    
    # Simulate class imbalance — real datasets are never perfectly balanced!
    # In CrisisMMD, "not_humanitarian" and "little_or_no_damage" are overrepresented
    humanitarian_weights = [0.10, 0.15, 0.12, 0.08, 0.20, 0.35]  # imbalanced
    damage_weights = [0.25, 0.35, 0.40]
    
    df = pd.DataFrame({
        "image_path": [f"data/raw/images/img_{i:04d}.jpg" for i in range(n)],
        "tweet_text": [f"Sample tweet text about disaster event {i}" for i in range(n)],
        "image_humanitarian": np.random.choice(
            humanitarian_labels, n, p=humanitarian_weights
        ),
        "image_damage": np.random.choice(
            damage_labels, n, p=damage_weights
        ),
        "disaster_name": np.random.choice(disaster_types, n),
        # Simulated image dimensions
        "img_width": np.random.choice([320, 480, 640, 1024], n),
        "img_height": np.random.choice([240, 360, 480, 768], n),
    })
    
    print(f"✅ Simulated dataset with {len(df)} samples")
    return df


# =============================================================================
# STEP 2: EDA FUNCTIONS
# =============================================================================

def plot_class_distribution(df: pd.DataFrame, output_dir: str = "outputs/eda"):
    """
    CONCEPT — Class Imbalance:
    In real disaster datasets, some categories are much rarer than others.
    For example, "severe_damage" images are far fewer than "no damage" images
    because most photos shared during disasters aren't of the worst destruction.
    
    WHY IT MATTERS:
    If your model trains on 800 "no damage" samples and only 50 "severe damage"
    samples, it will learn to predict "no damage" almost always — and still get
    high accuracy! This is misleading. We need to handle imbalance via:
      - Weighted loss functions
      - Oversampling minority classes (SMOTE for images)
      - Undersampling majority classes
    We'll address this in Week 1 Day 3 during training setup.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DisasterSense — Class Distribution Analysis", fontsize=14, fontweight="bold")
    
    # Plot 1: Humanitarian label distribution
    hum_counts = df["image_humanitarian"].value_counts()
    axes[0].barh(hum_counts.index, hum_counts.values, color=sns.color_palette("husl", len(hum_counts)))
    axes[0].set_title("Humanitarian Labels")
    axes[0].set_xlabel("Count")
    for i, v in enumerate(hum_counts.values):
        axes[0].text(v + 5, i, str(v), va="center", fontsize=9)
    
    # Plot 2: Damage severity distribution
    dmg_counts = df["image_damage"].value_counts()
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]  # red=severe, orange=mild, green=little
    axes[1].bar(dmg_counts.index, dmg_counts.values, color=colors)
    axes[1].set_title("Damage Severity Labels")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(dmg_counts.values):
        axes[1].text(i, v + 5, str(v), ha="center", fontsize=9)
    
    # Plot 3: Samples per disaster
    disaster_counts = df["disaster_name"].value_counts()
    axes[2].pie(
        disaster_counts.values,
        labels=[d.replace("_", "\n") for d in disaster_counts.index],
        autopct="%1.1f%%",
        colors=sns.color_palette("Set2", len(disaster_counts))
    )
    axes[2].set_title("Samples per Disaster Event")
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Saved class distribution plot → {save_path}")
    plt.show()


def plot_image_size_distribution(df: pd.DataFrame, output_dir: str = "outputs/eda"):
    """
    CONCEPT — Why image size matters:
    Neural networks require fixed-size inputs. EfficientNet-B0 expects 224x224.
    So we need to resize ALL images — but we need to know what sizes we're
    dealing with first.
    
    If most images are 640x480 and we resize to 224x224, we lose some detail
    but it's fine. If images are tiny (64x64), upscaling introduces blur.
    Knowing this informs our augmentation strategy.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Image Size Distribution", fontsize=13, fontweight="bold")
    
    axes[0].hist(df["img_width"], bins=20, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].axvline(224, color="red", linestyle="--", label="EfficientNet input (224)")
    axes[0].set_title("Image Widths")
    axes[0].set_xlabel("Width (px)")
    axes[0].legend()
    
    axes[1].hist(df["img_height"], bins=20, color="#9b59b6", edgecolor="white", alpha=0.8)
    axes[1].axvline(224, color="red", linestyle="--", label="EfficientNet input (224)")
    axes[1].set_title("Image Heights")
    axes[1].set_xlabel("Height (px)")
    axes[1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "image_sizes.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📊 Saved image size plot → {save_path}")
    plt.show()


def print_dataset_summary(df: pd.DataFrame):
    """
    CONCEPT — Always start with a summary.
    Before building any model, understand your data:
    - How many samples?
    - How many classes?
    - Are there missing values?
    - What does a sample look like?
    
    This is non-negotiable in real DS work. Garbage in = garbage out.
    """
    print("\n" + "="*60)
    print("📋 DATASET SUMMARY")
    print("="*60)
    print(f"Total samples     : {len(df):,}")
    print(f"Columns           : {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nHumanitarian label counts:\n{df['image_humanitarian'].value_counts()}")
    print(f"\nDamage label counts:\n{df['image_damage'].value_counts()}")
    print(f"\nDisaster event counts:\n{df['disaster_name'].value_counts()}")
    
    # Class imbalance ratio — important metric
    dmg_counts = df["image_damage"].value_counts()
    imbalance_ratio = dmg_counts.max() / dmg_counts.min()
    print(f"\n⚠️  Damage class imbalance ratio: {imbalance_ratio:.1f}x")
    if imbalance_ratio > 3:
        print("   → Significant imbalance detected! Will use weighted loss in training.")
    print("="*60)


def compute_label_encoding(df: pd.DataFrame) -> dict:
    """
    CONCEPT — Label Encoding:
    Neural networks work with numbers, not strings.
    We need to convert "severe_damage" → 0, "mild_damage" → 1, etc.
    
    We create a mapping dictionary and save it — IMPORTANT to save this
    so when we predict later, we can convert numbers back to class names.
    This mapping must be CONSISTENT between training and inference.
    """
    damage_classes = sorted(df["image_damage"].unique().tolist())
    humanitarian_classes = sorted(df["image_humanitarian"].unique().tolist())
    
    label_maps = {
        "damage": {label: idx for idx, label in enumerate(damage_classes)},
        "humanitarian": {label: idx for idx, label in enumerate(humanitarian_classes)},
        # Reverse maps for inference (number → class name)
        "damage_inv": {idx: label for idx, label in enumerate(damage_classes)},
        "humanitarian_inv": {idx: label for idx, label in enumerate(humanitarian_classes)},
    }
    
    print("\n🏷️  Label Encodings:")
    print(f"Damage     : {label_maps['damage']}")
    print(f"Humanitarian: {label_maps['humanitarian']}")
    
    return label_maps


# =============================================================================
# STEP 3: RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":
    
    print("🚀 DisasterSense — Day 1: Dataset EDA")
    print("-" * 40)
    
    # --- Load or simulate data ---
    # WHEN YOU HAVE REAL DATA: replace simulate_dataset() with:
    # df = load_crisisMMD_annotations("data/raw/CrisisMMD")
    df = simulate_dataset()
    
    # --- Summary ---
    print_dataset_summary(df)
    
    # --- Visualizations ---
    plot_class_distribution(df)
    plot_image_size_distribution(df)
    
    # --- Label encoding ---
    label_maps = compute_label_encoding(df)
    
    # --- Save processed dataframe ---
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/dataset_eda.csv", index=False)
    print("\n✅ Saved processed dataframe → data/processed/dataset_eda.csv")
    
    print("\n" + "="*60)
    print("✅ Day 1 Complete!")
    print("NEXT: Day 2 — Image preprocessing pipeline + train/val/test split")
    print("="*60)
