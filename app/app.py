# gradio app for the pidgin sentiment model
# the model is downloaded from the hugging face hub, the text is cleaned with the same
# clean_text function used in training
import os
import sys

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# on hugging face spaces preprocess.py sits next to this file, locally it's in ../src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_text

MODEL_ID = os.environ.get("MODEL_ID", "Divineitu/pidgin-sentiment")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()


def classify(text):
    if not text or not text.strip():
        return {}
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}


examples = [
    ["abeg this jollof sweet die, i go come back tomorrow"],
    ["na wa for this nepa o, light don go again since morning"],
    ["i dey go market later"],
    ["dis una network no dey work at all, wetin be this"],
]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Type a tweet in Nigerian Pidgin"),
    outputs=gr.Label(num_top_classes=3, label="Predicted sentiment"),
    examples=examples,
    title="Nigerian Pidgin Sentiment",
    description=(
        "Classifies a Nigerian Pidgin tweet as positive, negative or neutral. "
        "The model was fine-tuned on the AfriSenti Pidgin tweets. "
        "It struggles with neutral tweets because there were only 66 of them in training."
    ),
)

if __name__ == "__main__":
    demo.launch()
