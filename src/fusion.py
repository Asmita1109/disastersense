"""
DisasterSense | Fusion Layer
Combines image damage score and NLP informative score
into a single crisis severity score (0-100).
Auto-downloads models from HuggingFace Hub if not found locally.
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms
import torch.nn as nn

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_MODEL    = Path("models/image_model/best.pt")
NLP_MODEL      = Path("models/nlp_model/best")
NLP_NAME       = "cardiffnlp/twitter-roberta-base"
HF_REPO        = "AsmitaG11/disastersense-models"

DAMAGE_LABELS  = {0: "little_or_no_damage", 1: "mild_damage", 2: "severe_damage"}
DAMAGE_SCORES  = {0: 0.1, 1: 0.5, 2: 1.0}
IMAGE_WEIGHT   = 0.6
NLP_WEIGHT     = 0.4

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model Download ────────────────────────────────────────────────────────────

def ensure_models():
    from huggingface_hub import hf_hub_download

    if not IMAGE_MODEL.exists():
        print("Downloading image model...")
        IMAGE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=HF_REPO, filename="best.pt", local_dir="models/image_model")

    if not NLP_MODEL.exists():
        print("Downloading NLP model...")
        NLP_MODEL.mkdir(parents=True, exist_ok=True)
        for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
            hf_hub_download(repo_id=HF_REPO, filename=f, local_dir="models/nlp_model/best")


# ── Model Loaders ─────────────────────────────────────────────────────────────

def load_image_model():
    ensure_models()
    model = models.efficientnet_b0(weights=None)
    in_features      = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, 3),
    )
    model.load_state_dict(torch.load(IMAGE_MODEL, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def load_nlp_model():
    ensure_models()
    tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL)
    model.to(DEVICE).eval()
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_image(model, image_path: str) -> dict:
    img    = Image.open(image_path).convert("RGB")
    tensor = image_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().tolist()

    pred_idx     = int(torch.tensor(probs).argmax())
    damage_score = sum(DAMAGE_SCORES[i] * probs[i] for i in range(3))

    return {
        "predicted_class": DAMAGE_LABELS[pred_idx],
        "probabilities"  : {DAMAGE_LABELS[i]: round(probs[i], 4) for i in range(3)},
        "damage_score"   : round(damage_score, 4),
    }


def predict_text(model, tokenizer, text: str) -> dict:
    encoding = tokenizer(
        text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        probs = F.softmax(
            model(
                input_ids=encoding["input_ids"].to(DEVICE),
                attention_mask=encoding["attention_mask"].to(DEVICE)
            ).logits, dim=1
        ).squeeze().tolist()

    return {
        "predicted_class"  : "informative" if probs[1] > 0.5 else "not_informative",
        "informative_score": round(probs[1], 4),
        "probabilities"    : {"not_informative": round(probs[0], 4), "informative": round(probs[1], 4)},
    }


def compute_severity(image_result: dict, text_result: dict) -> dict:
    raw      = (IMAGE_WEIGHT * image_result["damage_score"]) + (NLP_WEIGHT * text_result["informative_score"])
    severity = round(raw * 100, 2)
    level    = "CRITICAL" if severity >= 70 else "HIGH" if severity >= 45 else "MODERATE" if severity >= 20 else "LOW"

    return {
        "severity_score"   : severity,
        "severity_level"   : level,
        "damage_score"     : image_result["damage_score"],
        "informative_score": text_result["informative_score"],
        "image_prediction" : image_result["predicted_class"],
        "text_prediction"  : text_result["predicted_class"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python fusion.py <image_path> '<tweet_text>'")
        sys.exit(1)

    image_model          = load_image_model()
    nlp_model, tokenizer = load_nlp_model()

    image_result = predict_image(image_model, sys.argv[1])
    text_result  = predict_text(nlp_model, tokenizer, sys.argv[2])
    severity     = compute_severity(image_result, text_result)

    print(f"\n── Image ─────────────────────────────────────────────")
    print(f"Class        : {image_result['predicted_class']}")
    print(f"Damage Score : {image_result['damage_score']}")
    print(f"\n── Text ──────────────────────────────────────────────")
    print(f"Class             : {text_result['predicted_class']}")
    print(f"Informative Score : {text_result['informative_score']}")
    print(f"\n── Severity ──────────────────────────────────────────")
    print(f"Score : {severity['severity_score']}/100")
    print(f"Level : {severity['severity_level']}")
