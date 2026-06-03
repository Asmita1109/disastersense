"""
DisasterSense | Gradio Demo
Standalone multimodal disaster severity detection with PostgreSQL logging.
"""

import sys
import os
import uuid
import random
import psycopg2
import gradio as gr
from datetime import datetime
from pathlib import Path

sys.path.append("src")
from fusion import load_image_model, load_nlp_model, predict_image, predict_text, compute_severity

print("Loading models...")
image_model          = load_image_model()
nlp_model, tokenizer = load_nlp_model()
print("Models loaded ✓")

# ── Example Scenarios ─────────────────────────────────────────────────────────

EXAMPLES = [
    {
        "image": "examples/severe1_new.jpg",
        "tweet": "Experts say Maria could hit Puerto Rico harder than Irma -- here's the latest on the storm",
        "label": "Severe Damage"
    },
    {
        "image": "examples/severe_2.jpg",
        "tweet": "California schools call off all sports because of rising smoke and ash from #CanyonFire2",
        "label": "Severe Damage"
    },
    {
        "image": "examples/mild_1.jpg",
        "tweet": "Some of the devastation from #Harvey friends and family who lost everything, prayers needed",
        "label": "Mild Damage"
    },
    {
        "image": "examples/mild_2.jpg",
        "tweet": "The indiscriminate fury of California's wildfires",
        "label": "Mild Damage"
    },
    {
        "image": "examples/low_1.jpg",
        "tweet": "Hurricane Maria update from Hato Rey Puerto Rico",
        "label": "Little/No Damage"
    },
    {
        "image": "examples/low_2.jpg",
        "tweet": "Tree down from tornado in plantation!",
        "label": "Little/No Damage"
    },
]


# ── Database Logging ──────────────────────────────────────────────────────────

def log_to_db(severity):
    try:
        conn = psycopg2.connect(
            host    =os.getenv("DB_HOST", "localhost"),
            port    =int(os.getenv("DB_PORT", 5432)),
            dbname  =os.getenv("DB_NAME", "disastersense"),
            user    =os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions (
                    prediction_id, timestamp, image_prediction, damage_score,
                    text_prediction, informative_score, severity_score,
                    severity_level, inference_time_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                uuid.uuid4().hex,
                datetime.utcnow().isoformat(),
                severity["image_prediction"],
                severity["damage_score"],
                severity["text_prediction"],
                severity["informative_score"],
                severity["severity_score"],
                severity["severity_level"],
                0.0,
            ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB logging failed: {e}")


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(image, tweet_text):
    if image is None:
        return "Please upload an image.", "", "", "", ""
    if not tweet_text.strip():
        return "Please enter tweet text.", "", "", "", ""

    try:
        image_result = predict_image(image_model, image)
        text_result  = predict_text(nlp_model, tokenizer, tweet_text)
        severity     = compute_severity(image_result, text_result)

        log_to_db(severity)

        level_icons = {
            "LOW"     : "🟢 LOW",
            "MODERATE": "🟡 MODERATE",
            "HIGH"    : "🟠 HIGH",
            "CRITICAL": "🔴 CRITICAL",
        }

        return (
            level_icons.get(severity["severity_level"], severity["severity_level"]),
            f"{severity['severity_score']}/100",
            severity["image_prediction"].replace("_", " ").title(),
            severity["text_prediction"].replace("_", " ").title(),
            f"{severity['damage_score']:.2f}",
        )

    except Exception as e:
        return f"Error: {str(e)}", "", "", "", ""


def load_random_example():
    example = random.choice(EXAMPLES)
    return example["image"], example["tweet"]


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="DisasterSense", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌍 DisasterSense
    ### Multimodal Disaster Severity Detection
    Upload a disaster image and paste a related tweet — or try a random example!
    > ⚠️ First prediction may take 30-60 seconds while models load.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Disaster Image")
            text_input  = gr.Textbox(
                lines=3,
                placeholder="Paste a disaster-related tweet here...",
                label="Tweet Text"
            )
            with gr.Row():
                example_btn = gr.Button("🎲 Try a Random Example", variant="secondary")
                submit_btn  = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            severity_level = gr.Textbox(label="Severity Level", interactive=False)
            severity_score = gr.Textbox(label="Severity Score (0-100)", interactive=False)
            image_pred     = gr.Textbox(label="Image Prediction", interactive=False)
            text_pred      = gr.Textbox(label="Text Prediction", interactive=False)
            damage_score   = gr.Textbox(label="Damage Score", interactive=False)

    gr.Markdown("""
    ---
    **Model Details:**
    - Image Classifier: EfficientNet-B0 fine-tuned on CrisisMMD v2.0 (64% accuracy)
    - NLP Classifier: twitter-roberta-base fine-tuned on CrisisMMD v2.0 (75% accuracy)
    - Fusion: Weighted combination (60% image, 40% text)
    - Dataset: 7 real disaster events — Harvey, Irma, Maria, California Wildfires and more

    **[[GitHub]](https://github.com/Asmita1109/disastersense)**
    """)

    example_btn.click(
        fn=load_random_example,
        inputs=[],
        outputs=[image_input, text_input],
    )

    submit_btn.click(
        fn=predict,
        inputs=[image_input, text_input],
        outputs=[severity_level, severity_score, image_pred, text_pred, damage_score],
    )

if __name__ == "__main__":
    demo.launch()
