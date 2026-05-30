"""
DisasterSense | NLP Preprocessing
Tokenization and dataloaders for twitter-roberta-base
fine-tuning on CrisisMMD informative classification task.
"""

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

RAW_DIR   = Path("data/raw/crisismmd_datasplit_all")
PROCESSED = Path("data/processed")
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
MAX_LEN    = 128
LABEL_MAP  = {"informative": 1, "not_informative": 0}


def load_splits():
    splits = {}
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(RAW_DIR / f"task_informative_text_img_{split}.tsv", sep="\t")
        df.columns = df.columns.str.strip()
        df["tweet_text"] = df["tweet_text"].astype(str).str.strip()
        df = df.dropna(subset=["tweet_text", "label"])
        splits[split] = df
        print(f"{split:6s} → {len(df):,} samples | labels: {df['label'].value_counts().to_dict()}")
    return splits


class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts     = df["tweet_text"].tolist()
        self.labels    = [LABEL_MAP[l] for l in df["label"]]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"     : encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_nlp_dataloaders(batch_size=32):
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    splits  = load_splits()
    loaders = {}

    for split, df in splits.items():
        ds             = TweetDataset(df, tokenizer)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        print(f"{split:6s} loader → {len(ds):,} samples | {len(loaders[split])} batches")

    return loaders, tokenizer


def verify_batch(loaders):
    batch = next(iter(loaders["train"]))
    print(f"\nBatch keys     : {list(batch.keys())}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"Labels         : {batch['label'][:8]}")
    assert batch["input_ids"].shape[1] == MAX_LEN
    print("Sanity checks passed ✓")


if __name__ == "__main__":
    print("── NLP Preprocessing ─────────────────────────────────")
    loaders, tokenizer = build_nlp_dataloaders()

    print("\n── Batch Verification ────────────────────────────────")
    verify_batch(loaders)

    # Save processed splits
    splits = load_splits()
    for split, df in splits.items():
        df.to_csv(PROCESSED / f"informative_{split}.csv", index=False)
    print("\nSaved → data/processed/informative_*.csv")
