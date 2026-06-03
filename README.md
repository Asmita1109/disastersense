# 🌍 DisasterSense
### Multimodal Disaster Detection & Severity Scoring System

A production-ready AI system that classifies disaster images and social media text, fuses both signals into a crisis severity score, and monitors predictions on a live dashboard.

---

## 🔴 Live Links

| | |
|---|---|
| **Interactive Demo** | [huggingface.co/spaces/AsmitaG11/disastersense](https://huggingface.co/spaces/AsmitaG11/disastersense) |
| **Live Dashboard** | [Crisis Monitoring Dashboard](http://metabase-production-f613.up.railway.app/public/dashboard/8a13a7dd-61e6-4155-92c1-f73be0565b7a) |

---

## Demo

![DisasterSense Demo](outputs/demo/Gradio_UI_demo.png)

**Severity: 84.8/100 — 🔴 CRITICAL** | Image: Severe Damage | Text: Informative

---

## Architecture

```
[Disaster Image] ──→ [EfficientNet-B0] ──→ damage_score
                                                  ↓
                                          [Fusion Layer] ──→ Severity Score (0–100)
                                                  ↑
[Tweet Text]     ──→ [twitter-roberta] ──→ informative_score

                              ↓
                    ┌─────────────────────┐
                    │     FastAPI          │  ← Local REST API
                    └─────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │  Railway PostgreSQL  │  ← Cloud database (single source of truth)
                    └─────────────────────┘
                         ↑           ↑
              HuggingFace Space    Local API
              (public demo)        (development)
                              ↓
                    ┌─────────────────────┐
                    │   Metabase Cloud     │  ← Live public dashboard
                    └─────────────────────┘
```

---

## Results

| Model | Task | Accuracy |
|---|---|---|
| EfficientNet-B0 | Damage Severity (3-class) | 64% |
| twitter-roberta-base | Informative Classification | 75% |

### Model Improvement
| Version | Val Accuracy | Changes |
|---|---|---|
| Baseline | 58.22% | Frozen backbone, 15 epochs |
| Improved | 64.27% | Unfrozen last 3 blocks, differential LR, 20 epochs |

---

## Tech Stack

| Component | Tool |
|---|---|
| Image Classifier | EfficientNet-B0 (PyTorch) |
| NLP Classifier | twitter-roberta-base (HuggingFace) |
| Fusion Layer | Weighted scoring (60% image, 40% text) |
| API | FastAPI + Uvicorn |
| Database | Railway PostgreSQL (cloud) |
| Dashboard | Metabase (Railway) |
| Demo UI | Gradio (HuggingFace Spaces) |
| Containerization | Docker + docker-compose |

---

## Cloud Infrastructure

```
┌─────────────────────────────────────────────────────┐
│                 HuggingFace Spaces                   │
│  Gradio UI + EfficientNet-B0 + twitter-roberta-base  │
│  CPU Basic (16GB RAM) — Free tier                    │
└───────────────────────┬─────────────────────────────┘
                        │ logs predictions
                        ↓
┌─────────────────────────────────────────────────────┐
│                 Railway PostgreSQL                   │
│  Stores all predictions — always online             │
│  Accessible from HuggingFace + local API            │
└───────────────────────┬─────────────────────────────┘
                        │ reads data
                        ↓
┌─────────────────────────────────────────────────────┐
│                 Metabase on Railway                  │
│  Public live dashboard — auto-refreshes every 5 min │
│  http://metabase-production-f613.up.railway.app      │
└─────────────────────────────────────────────────────┘
```

---

## Dataset

**CrisisMMD v2.0** (Alam et al., 2018)
- 7 real disaster events: Hurricane Harvey, Irma, Maria, Iraq-Iran Earthquake, Mexico Earthquake, Sri Lanka Floods, California Wildfires
- 3,526 labeled images for damage severity classification
- 13,608 labeled tweets for informative classification

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Model + DB health check |
| POST | `/predict` | Multimodal severity prediction |
| GET | `/predictions` | Last 50 logged predictions |
| GET | `/labels` | Available classes and thresholds |

### Sample Response
```json
{
  "prediction_id": "67f9543bf6514866a6a36c52d08aba1d",
  "timestamp": "2026-05-31T03:30:42.910005",
  "image_prediction": "severe_damage",
  "damage_score": 0.8302,
  "text_prediction": "informative",
  "informative_score": 0.6928,
  "severity_score": 77.52,
  "severity_level": "CRITICAL",
  "inference_time_ms": 198.04
}
```

---

## Severity Levels

| Level | Score | Meaning |
|---|---|---|
| 🟢 LOW | 0–20 | Minimal damage, low social signal |
| 🟡 MODERATE | 20–45 | Some damage detected |
| 🟠 HIGH | 45–70 | Significant damage, active reporting |
| 🔴 CRITICAL | 70–100 | Severe damage, high social activity |

---

## Running Locally

```bash
git clone https://github.com/Asmita1109/disastersense.git
cd disastersense
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Create .env file with your DB credentials
echo "DB_HOST=your_railway_host" > .env
echo "DB_PORT=your_railway_port" >> .env
echo "DB_NAME=railway" >> .env
echo "DB_USER=postgres" >> .env
echo "DB_PASSWORD=your_password" >> .env

# Start API
uvicorn api.main:app --reload

# Start demo UI (separate terminal)
python app.py
```

### Docker
```bash
docker-compose up
```

---

## Project Structure

```
disastersense/
├── notebooks/
│   └── eda.py                  # Exploratory data analysis
├── src/
│   ├── preprocess.py           # Image transforms and dataloaders
│   ├── train.py                # EfficientNet-B0 fine-tuning
│   ├── nlp_preprocess.py       # Tweet tokenization
│   ├── nlp_train.py            # twitter-roberta fine-tuning
│   ├── fusion.py               # Multimodal fusion layer
│   ├── batch_predict.py        # Batch inference script
│   └── migrate_to_railway.py   # ETL migration script
├── api/
│   └── main.py                 # FastAPI application
├── examples/                   # Demo images for Gradio UI
├── outputs/
│   ├── eda/                    # EDA charts and model results
│   ├── api/                    # API response screenshots
│   ├── dashboard/              # Metabase dashboard export
│   └── demo/                   # Gradio demo screenshots
├── app.py                      # Gradio demo interface
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Known Limitations

- NLP model shows slight bias toward `informative` class due to class imbalance (8,341 vs 5,267 samples)
- Image model trained on CPU — higher accuracy achievable with GPU fine-tuning
- 72% of training data from hurricane events — model generalizes less well to earthquake/flood imagery
- Docker deployment requires memory-optimized hosting (>512MB RAM) for concurrent PyTorch model loading

## Future Scope

- Weighted cross-entropy loss to reduce NLP model bias
- Real-time Twitter/X stream ingestion via Tweepy or Kafka
- Full EfficientNet fine-tuning on GPU for higher accuracy
- Geographic clustering of predictions on a live map
- Multi-language support for global disaster response
- Active learning pipeline for continuous model improvement
- Docker cloud deployment on memory-optimized hosting

---

## References

Alam, F., Ofli, F., & Imran, M. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. *ICWSM 2018*.
