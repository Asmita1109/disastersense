"""
DisasterSense | FastAPI Application
REST API for multimodal disaster severity prediction.
Logs every prediction to PostgreSQL.
"""
import os
import sys
import uuid
import time
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.append(str(Path(__file__).parent.parent / "src"))
from fusion import load_image_model, load_nlp_model, predict_image, predict_text, compute_severity

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "DisasterSense API",
    description = "Multimodal disaster detection and severity scoring",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database ──────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host"    : os.getenv("PGHOST"),
    "port"    : int(os.getenv("PGPORT")),
    "dbname"  : os.getenv("DB_NAME", "railway"),
    "user"    : os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
}

def get_db():
    return psycopg2.connect(**DB_CONFIG)

def log_prediction(data: dict):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions (
                    prediction_id, timestamp, image_prediction, damage_score,
                    text_prediction, informative_score, severity_score,
                    severity_level, inference_time_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                data["prediction_id"], data["timestamp"],
                data["image_prediction"], data["damage_score"],
                data["text_prediction"], data["informative_score"],
                data["severity_score"], data["severity_level"],
                data["inference_time_ms"],
            ))
        conn.commit()
    finally:
        conn.close()


# ── Models ────────────────────────────────────────────────────────────────────

image_model          = None
nlp_model, tokenizer = None, None

@app.on_event("startup")
async def load_models():
    print("API started — models will load on first request")


# ── Schema ────────────────────────────────────────────────────────────────────

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
    global image_model, nlp_model, tokenizer
    if image_model is None:
        image_model = load_image_model()
    if nlp_model is None:
        nlp_model, tokenizer = load_nlp_model()

    start = time.time()

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    tmp_path = Path(f"tmp_{uuid.uuid4().hex}.jpg")
    try:
        tmp_path.write_bytes(await image.read())
        image_result = predict_image(image_model, str(tmp_path))
        text_result  = predict_text(nlp_model, tokenizer, text)
        severity     = compute_severity(image_result, text_result)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    inference_ms = round((time.time() - start) * 1000, 2)

    response = SeverityResponse(
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

    log_prediction(response.dict())
    return response


@app.get("/predictions")
def get_predictions(limit: int = 50):
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT %s", (limit,))
            return cur.fetchall()
    finally:
        conn.close()


@app.get("/labels")
def get_labels():
    return {
        "damage_classes"     : ["little_or_no_damage", "mild_damage", "severe_damage"],
        "informative_classes": ["informative", "not_informative"],
        "severity_levels"    : ["LOW", "MODERATE", "HIGH", "CRITICAL"],
        "severity_thresholds": {"LOW": "0-20", "MODERATE": "20-45", "HIGH": "45-70", "CRITICAL": "70-100"},
    }
