# 🌍 DisasterSense
### Multimodal Disaster Detection & Severity Scoring System

> Classifies crisis images and social media text, fuses them into a real-time severity score, served via REST API and monitored on a live Metabase dashboard.

---

## Architecture

```
[Image Input] ──→ [EfficientNet-B0] ──→ image_score
                                              ↓
                                      [Fusion Layer] ──→ Crisis Severity Score (0–100)
                                              ↑
[Text Input]  ──→ [RoBERTa]         ──→ text_score
                        ↓
                   [FastAPI]
                        ↓
                  [PostgreSQL]
                        ↓
                  [Metabase Dashboard]
```

## Stack
| Component | Tool |
|---|---|
| Image Classification | EfficientNet-B0 (PyTorch) |
| NLP Classification | twitter-roberta-base (HuggingFace) |
| API | FastAPI |
| Database | PostgreSQL |
| Dashboard | Metabase |
| UI | Gradio |
| Deploy | Render + HuggingFace Spaces |

## Dataset
- **Images**: CrisisMMD (7 disaster events, ~18k images)
- **Text**: HumAID (77k disaster tweets, 11 categories)

## Week-by-Week Build Plan
- **Week 1**: EDA + Image classifier
- **Week 2**: NLP classifier  
- **Week 3**: Fusion + FastAPI + PostgreSQL + Metabase
- **Week 4**: Deploy + Gradio UI + Documentation

## Running Locally
```bash
pip install -r requirements.txt
python notebooks/day1_eda.py
```

## Results
*To be updated as project progresses*

---
Built as a portfolio project combining Computer Vision, NLP, and MLOps.
