# Converts the fine-tuned model to ONNX so it can run in the browser with transformers.js.
# HF only hosts Gradio Spaces on PRO now, but static (HTML) Spaces are free, so the app runs client side.
# Run from the app folder after `hf auth login`:  python export_onnx.py
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import HfApi, hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.float16 import convert_float_to_float16
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "DivineItu/pidgin-sentiment"
OUT = "../models/onnx_export"
os.makedirs(f"{OUT}/onnx", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
model.config.return_dict = False

sample = tokenizer(["abeg this jollof sweet die"], return_tensors="pt")
torch.onnx.export(
    model,
    (sample["input_ids"], sample["attention_mask"]),
    f"{OUT}/onnx/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                  "logits": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)

# smaller versions so the page loads faster
# int8 is 4x smaller but changed ~12% of the labels on 600 test tweets (acc 0.650 -> 0.632)
# fp16 is 2x smaller and gave exactly the same labels, so the app uses fp16
quantize_dynamic(f"{OUT}/onnx/model.onnx", f"{OUT}/onnx/model_quantized.onnx", weight_type=QuantType.QInt8)
fp16 = convert_float_to_float16(onnx.load(f"{OUT}/onnx/model.onnx"), keep_io_types=True)
onnx.save(fp16, f"{OUT}/onnx/model_fp16.onnx")

# check the onnx versions give (almost) the same answers as pytorch
texts = ["abeg this jollof sweet die", "dis una network no dey work at all", "wahala dey o",
         "no wham na, make i dey go", "massive love"]
enc = tokenizer(texts, padding=True, return_tensors="pt")
with torch.no_grad():
    torch_probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"])[0], dim=-1).numpy()
for name in ["model.onnx", "model_quantized.onnx", "model_fp16.onnx"]:
    sess = ort.InferenceSession(f"{OUT}/onnx/{name}")
    logits = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    same = (probs.argmax(1) == torch_probs.argmax(1)).mean()
    print(f"{name}: {os.path.getsize(f'{OUT}/onnx/{name}') / 1e6:.0f} MB, "
          f"max prob diff {np.abs(probs - torch_probs).max():.4f}, same label {same:.0%}")

api = HfApi()
api.upload_folder(folder_path=f"{OUT}/onnx", path_in_repo="onnx", repo_id=MODEL_ID,
                  commit_message="adding onnx version for the browser app")

# transformers v5 saves tokenizer.json in a newer format that transformers.js 3.x reads wrong
# (it drops the space marker before each word, so "dis" became a different token and the predictions changed).
# the base model's tokenizer.json has the same vocab in the old format and gave identical ids on all
# 3,228 clean test tweets, so it replaces ours
old_tok = hf_hub_download("Davlan/afro-xlmr-base", "tokenizer.json")
api.upload_file(path_or_fileobj=old_tok, path_in_repo="tokenizer.json", repo_id=MODEL_ID,
                commit_message="tokenizer.json in the format transformers.js can read")
print("uploaded onnx files to", MODEL_ID)
