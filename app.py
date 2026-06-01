"""
DisasterSense | Gradio Demo Interface
Interactive UI for multimodal disaster severity prediction.
"""

import sys
import requests
import gradio as gr
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict"


def predict(image, tweet_text):
    if image is None:
        return "Please upload an image.", "", "", "", ""

    if not tweet_text.strip():
        return "Please enter tweet text.", "", "", "", ""

    try:
        with open(image, "rb") as f:
            response = requests.post(
                API_URL,
                files={"image": ("image.jpg", f, "image/jpeg")},
                data={"text": tweet_text},
                timeout=30,
            )

        if response.status_code != 200:
            return f"API Error: {response.status_code}", "", "", "", ""

        data = response.json()

        severity_score  = f"{data['severity_score']}/100"
        severity_level  = data["severity_level"]
        image_pred      = data["image_prediction"].replace("_", " ").title()
        text_pred       = data["text_prediction"].replace("_", " ").title()
        damage_score    = f"{data['damage_score']:.2f}"

        level_colors = {
            "LOW"     : "🟢 LOW",
            "MODERATE": "🟡 MODERATE",
            "HIGH"    : "🟠 HIGH",
            "CRITICAL": "🔴 CRITICAL",
        }

        return (
            level_colors.get(severity_level, severity_level),
            severity_score,
            image_pred,
            text_pred,
            damage_score,
        )

    except requests.exceptions.ConnectionError:
        return "API not running. Start with: uvicorn api.main:app --reload", "", "", "", ""
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", ""


with gr.Blocks(title="DisasterSense", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌍 DisasterSense
    ### Multimodal Disaster Severity Detection
    Upload a disaster image and paste a related tweet to get a real-time crisis severity score.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Disaster Image")
            text_input  = gr.Textbox(
                lines=3,
                placeholder="Paste a disaster-related tweet here...",
                label="Tweet Text"
            )
            submit_btn  = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            severity_level  = gr.Textbox(label="Severity Level", interactive=False)
            severity_score  = gr.Textbox(label="Severity Score (0-100)", interactive=False)
            image_pred      = gr.Textbox(label="Image Prediction", interactive=False)
            text_pred       = gr.Textbox(label="Text Prediction", interactive=False)
            damage_score    = gr.Textbox(label="Damage Score", interactive=False)

    gr.Markdown("""
    ---
    **Model Details:**
    - Image Classifier: EfficientNet-B0 fine-tuned on CrisisMMD v2.0 (64% accuracy)
    - NLP Classifier: twitter-roberta-base fine-tuned on CrisisMMD v2.0 (75% accuracy)
    - Fusion: Weighted combination (60% image, 40% text)
    - Dataset: 7 real disaster events — Harvey, Irma, Maria, California Wildfires, and more
    """)

    submit_btn.click(
        fn=predict,
        inputs=[image_input, text_input],
        outputs=[severity_level, severity_score, image_pred, text_pred, damage_score],
    )

if __name__ == "__main__":
    demo.launch(share=False)
