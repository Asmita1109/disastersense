"""
DisasterSense | Batch Prediction
Runs predictions on all test set images and logs them to PostgreSQL.
"""

import sys
import uuid
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

API_URL   = "http://127.0.0.1:8000/predict"
TEST_CSV  = Path("data/processed/damage_test.csv")
IMAGE_DIR = Path("data/raw/CrisisMMD_v2.0")


def run_batch():
    df = pd.read_csv(TEST_CSV)
    print(f"Running predictions on {len(df)} test images...")

    success, failed = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = IMAGE_DIR / row["image"]

        if not img_path.exists():
            failed += 1
            continue

        try:
            with open(img_path, "rb") as f:
                response = requests.post(
                    API_URL,
                    files={"image": (img_path.name, f, "image/jpeg")},
                    data={"text": str(row["tweet_text"])},
                    timeout=30,
                )

            if response.status_code == 200:
                success += 1
            else:
                failed += 1

        except Exception as e:
            failed += 1
            continue

    print(f"\nDone! Success: {success} | Failed: {failed}")
    print(f"Total predictions logged to database: {success}")


if __name__ == "__main__":
    run_batch()
