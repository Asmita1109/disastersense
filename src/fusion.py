"""
DisasterSense | Fusion Layer
Combines image damage score and NLP informative score
into a single crisis severity score (0-100).
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms
import torch.nn as nn

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_MODEL = Path("models/image_model/best.pt")
NLP_MODEL   = Path("models/nlp_model/best")
NLP_NAME    = "cardiffnlp/twitter-roberta-base"

DAMAGE_LABELS = {0: "little_or_no_damage", 1: "mild_damage", 2: "severe_damage"}
DAMAGE_SCORES = {0: 0.1, 1: 0.5, 2: 1.0}  # damage severity weights

IMAGE_WEIGHT  = 0.6
NLP_WEIGHT    = 0.4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model Loaders ─────────────────────────────────────────────────────────────

def load_image_model():
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
    tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL)
    model.to(DEVICE).eval()
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_image(model, image_path: str) -> dict:
    img    = Image.open(image_path).convert("RGB")
    tensor = image_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().tolist()

    pred_idx   = int(torch.tensor(probs).argmax())
    pred_label = DAMAGE_LABELS[pred_idx]
    damage_score = sum(DAMAGE_SCORES[i] * probs[i] for i in range(3))

    return {
        "predicted_class" : pred_label,
        "probabilities"   : {DAMAGE_LABELS[i]: round(probs[i], 4) for i in range(3)},
        "damage_score"    : round(damage_score, 4),
    }


def predict_text(model, tokenizer, text: str) -> dict:
    encoding = tokenizer(
        text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = F.softmax(logits, dim=1).squeeze().tolist()

    informative_score = probs[1]  # probability of being informative

    return {
        "predicted_class"  : "informative" if informative_score > 0.5 else "not_informative",
        "informative_score": round(informative_score, 4),
        "probabilities"    : {
            "not_informative": round(probs[0], 4),
            "informative"    : round(probs[1], 4),
        },
    }


def compute_severity(image_result: dict, text_result: dict) -> dict:
    damage_score     = image_result["damage_score"]
    informative_score = text_result["informative_score"]

    raw_score    = (IMAGE_WEIGHT * damage_score) + (NLP_WEIGHT * informative_score)
    severity     = round(raw_score * 100, 2)

    if severity >= 70:
        level = "CRITICAL"
    elif severity >= 45:
        level = "HIGH"
    elif severity >= 20:
        level = "MODERATE"
    else:
        level = "LOW"

    return {
        "severity_score"   : severity,
        "severity_level"   : level,
        "damage_score"     : damage_score,
        "informative_score": informative_score,
        "image_prediction" : image_result["predicted_class"],
        "text_prediction"  : text_result["predicted_class"],
    }


def predict(image_path: str, text: str) -> dict:
    image_model            = load_image_model()
    nlp_model, tokenizer   = load_nlp_model()

    image_result = predict_image(image_model, image_path)
    text_result  = predict_text(nlp_model, tokenizer, text)
    severity     = compute_severity(image_result, text_result)

    return {
        "image"   : image_result,
        "text"    : text_result,
        "severity": severity,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python fusion.py <image_path> '<tweet_text>'")
        sys.exit(1)

    image_path = sys.argv[1]
    tweet_text = sys.argv[2]

    print(f"\nImage : {image_path}")
    print(f"Text  : {tweet_text}\n")

    result = predict(image_path, tweet_text)

    print(f"── Image Prediction ──────────────────────────────────")
    print(f"Class        : {result['image']['predicted_class']}")
    print(f"Damage Score : {result['image']['damage_score']}")
    print(f"Probabilities: {result['image']['probabilities']}")

    print(f"\n── Text Prediction ───────────────────────────────────")
    print(f"Class            : {result['text']['predicted_class']}")
    print(f"Informative Score: {result['text']['informative_score']}")

    print(f"\n── Crisis Severity ───────────────────────────────────")
    print(f"Score : {result['severity']['severity_score']}/100")
    print(f"Level : {result['severity']['severity_level']}")
