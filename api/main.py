"""
DisasterSense | FastAPI Application
REST API for multimodal disaster severity prediction.
"""

import sys
import uuid
import time
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

sys.path.append(str(Path(__file__).parent.parent / "src"))
from fusion import predict, load_image_model, load_nlp_model

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "DisasterSense API",
    description = "Multimodal disaster detection and severity scoring",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Load Models on Startup ────────────────────────────────────────────────────

image_model          = None
nlp_model, tokenizer = None, None

@app.on_event("startup")
async def load_models():
    global image_model, nlp_model, tokenizer
    print("Loading models...")
    image_model          = load_image_model()
    nlp_model, tokenizer = load_nlp_model()
    print("Models loaded ✓")


# ── Response Schema ───────────────────────────────────────────────────────────

class SeverityResponse(BaseModel):
    prediction_id     : str
    timestamp         : str
    image_prediction  : str
    damage_score      : float
    text_prediction   : str
    informative_score : float
    severity_score    : float
    severity_level    : str
    inference_time_ms : float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "DisasterSense API", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status"      : "healthy",
        "image_model" : image_model is not None,
        "nlp_model"   : nlp_model is not None,
        "device"      : str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }


@app.post("/predict", response_model=SeverityResponse)
async def predict_severity(
    image: UploadFile = File(...),
    text : str        = Form(...),
):
    start = time.time()

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    tmp_path = Path(f"tmp_{uuid.uuid4().hex}.jpg")
    try:
        contents = await image.read()
        tmp_path.write_bytes(contents)

        from fusion import predict_image, predict_text, compute_severity
        image_result = predict_image(image_model, str(tmp_path))
        text_result  = predict_text(nlp_model, tokenizer, text)
        severity     = compute_severity(image_result, text_result)

    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    inference_ms = round((time.time() - start) * 1000, 2)

    return SeverityResponse(
        prediction_id     = uuid.uuid4().hex,
        timestamp         = datetime.utcnow().isoformat(),
        image_prediction  = severity["image_prediction"],
        damage_score      = severity["damage_score"],
        text_prediction   = severity["text_prediction"],
        informative_score = severity["informative_score"],
        severity_score    = severity["severity_score"],
        severity_level    = severity["severity_level"],
        inference_time_ms = inference_ms,
    )


@app.get("/labels")
def get_labels():
    return {
        "damage_classes"     : ["little_or_no_damage", "mild_damage", "severe_damage"],
        "informative_classes": ["informative", "not_informative"],
        "severity_levels"    : ["LOW", "MODERATE", "HIGH", "CRITICAL"],
        "severity_thresholds": {"LOW": "0-20", "MODERATE": "20-45", "HIGH": "45-70", "CRITICAL": "70-100"},
    }
