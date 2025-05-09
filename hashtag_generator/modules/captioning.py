import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import re
from typing import Tuple

from utils.file_io import save_result, copy_image

def load_img_to_text_model(model_name: str, torch_dtype=torch.float16, device_map="auto"):
    processor = AutoProcessor.from_pretrained(model_name)
    model     = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    return processor, model

def generate_caption(
    image_path: str,
    processor,
    model,
    prompt: str,
    device: torch.device
) -> str:
    img = Image.open(image_path).convert("RGB")
    formatted = f"<|system|>You are a helpful assistant.<|user|><image>{prompt}<|assistant|>"
    inputs = processor(text=formatted, images=img, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=64,
    )
    raw_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if "<|assistant|>" in raw_text:
        text = text.split("<|assistant|>")[-1].strip()
    return raw_text, text

def process_captioning(
    json_dir: str,
    output_dir: str,
    model_name: str,
    prompt: str
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_img_to_text_model(model_name)

    os.makedirs(output_dir, exist_ok=True)
    for label in ("pass", "hold", "non-pass"):
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    for fname in os.listdir(json_dir):
        if not fname.lower().endswith(".json"):
            continue

        print(f'caption {fname}')

        json_path = os.path.join(json_dir, fname)
        rec = json.load(open(json_path, encoding="utf-8"))

        clip_decision = rec.get('clip_decision')
        if clip_decision is not None and clip_decision not in ("pass", "hold"):
            continue

        img_path = rec.get("file_path")
        if not img_path or not os.path.isfile(img_path):
            print(f"[이미지 없음] {json_path}")
            continue

        raw_caption, caption = generate_caption(img_path, processor, model, prompt, device)

        rec.update({
            "caption": caption,
            "raw_caption": raw_caption
        })

        save_result(rec, json_path)