---
title: Nigerian Pidgin Sentiment
emoji: 💬
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
---

# Nigerian Pidgin Sentiment

Type a tweet in Nigerian Pidgin and get positive / negative / neutral.

The model is AfroXLMR-base fine-tuned on the AfriSenti Nigerian Pidgin tweets. It runs in the browser with
transformers.js (ONNX, fp16), so the page downloads the model once (~550 MB) and then works offline.
The training data is AfriSenti (CC BY-NC-SA), so this is for non-commercial use only.
Code, data preparation, experiments and results: https://github.com/Divineitu/pidgin-sentiment
