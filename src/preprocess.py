"""
DisasterSense | Image Preprocessing
Transforms, dataset class, class weights, and dataloaders
for EfficientNet-B0 fine-tuning on CrisisMMD damage severity.
"""

import os
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_BASE    = Path("data/raw/CrisisMMD_v2.0")
PROCESSED     = Path("data/processed")
LABEL_MAP     = {"little_or_no_damage": 0, "mild_damage": 1, "severe_damage": 2}
NUM_CLASSES   = len(LABEL_MAP)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class CrisisDataset(Dataset):
    def __init__(self, csv_path, image_base, transform=None):
        self.df         = pd.read_csv(csv_path)
        self.image_base = image_base
        self.transform  = transform
        self._drop_missing()

    def _drop_missing(self):
        valid   = self.df["image"].apply(lambda p: (self.image_base / p).exists())
        dropped = (~valid).sum()
        if dropped:
            print(f"Dropped {dropped} rows with missing images.")
        self.df = self.df[valid].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.image_base / row["image"]).convert("RGB")
        label = LABEL_MAP[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


def compute_class_weights(csv_path):
    df      = pd.read_csv(csv_path)
    counts  = Counter(df["label"])
    total   = sum(counts.values())
    weights = []
    for label in sorted(LABEL_MAP.keys()):
        w = total / (NUM_CLASSES * counts[label])
        weights.append(w)
        print(f"  {label:25s} → {counts[label]:4d} samples | weight: {w:.3f}")
    return torch.tensor(weights, dtype=torch.float)


def build_dataloaders(batch_size=32):
    splits = {
        "train": (PROCESSED / "damage_train.csv", train_transforms),
        "dev"  : (PROCESSED / "damage_dev.csv",   eval_transforms),
        "test" : (PROCESSED / "damage_test.csv",   eval_transforms),
    }
    loaders = {}
    for split, (csv, tfm) in splits.items():
        ds             = CrisisDataset(csv, IMAGE_BASE, transform=tfm)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"{split:6s} → {len(ds):,} samples | {len(loaders[split])} batches")
    return loaders


def verify_batch(loaders):
    images, labels = next(iter(loaders["train"]))
    print(f"Batch shape  : {images.shape}")
    print(f"Pixel range  : [{images.min():.3f}, {images.max():.3f}]")
    assert images.shape[1:] == (3, 224, 224)
    assert -3.0 <= images.min() and images.max() <= 3.0
    print("Sanity checks passed ✓")


if __name__ == "__main__":
    print("── Class Weights ─────────────────────────────────────")
    weights = compute_class_weights(PROCESSED / "damage_train.csv")
    print(f"\nWeights: {weights}")

    print("\n── DataLoaders ───────────────────────────────────────")
    loaders = build_dataloaders()

    print("\n── Verification ──────────────────────────────────────")
    verify_batch(loaders)

    os.makedirs("models", exist_ok=True)
    torch.save(weights, "models/class_weights.pt")
    print("Saved → models/class_weights.pt")
