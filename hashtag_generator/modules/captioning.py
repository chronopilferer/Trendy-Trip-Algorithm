import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
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
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text

def parse_fine_caption(caption: str) -> Tuple[str, str, str]:
    indoor_out = place = full_env = None
    for line in caption.splitlines():
        line = line.strip()
        if line.lower().startswith("1"):
            indoor_out = line.split("?",1)[-1].strip().split()[0]
        elif line.lower().startswith("2"):
            place = line.split("?",1)[-1].strip().rstrip(".")
        elif line.lower().startswith("3"):
            full_env = line.split("?",1)[-1].strip().split()[0]
    return indoor_out, place, full_env

def process_fine_captioning(
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

        json_path = os.path.join(json_dir, fname)
        rec = json.load(open(json_path, encoding="utf-8"))

        if not rec.get("pass", False):
            continue

        img_path = rec.get("filepath")
        if not img_path or not os.path.isfile(img_path):
            print(f"[이미지 없음] {json_path}")
            continue

        caption = generate_caption(img_path, processor, model, prompt, device)
        indoor_out, place, full_env = parse_fine_caption(caption)

        rec.update({
            "caption": caption,
            "indoor_or_outdoor": indoor_out,
            "place_type": place,
            "full_environment_visible": full_env
        })

        fine_decision = "pass" if full_env and full_env.lower().startswith("yes") else "hold"
        rec["fine_decision"] = fine_decision

        rec["pass"] = True if fine_decision in ("pass", "hold") else False

        save_result(rec, json_path)

        if fine_decision in ("pass", "hold"):
            copy_image(img_path, output_dir, fine_decision)
            print(f"[{fine_decision.upper()}] {fname}: {indoor_out}/{place}/{full_env}")
        else:
            print(f"[SKIP NON-PASS] {fname}")

