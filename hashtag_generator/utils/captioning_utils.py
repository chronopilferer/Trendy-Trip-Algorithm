import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from captioning.caption_generator import generate_caption
from utils.config_loader import load_config
import os
import json

def load_blip_model(model_name):
    print(f"[Loading Model] {model_name}")
    processor = InstructBlipProcessor.from_pretrained(model_name)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

def create_caption_result(filename: str, image_path: str, caption: str) -> dict:
    return {
        "filename": filename,
        "filepath": image_path,
        "caption": caption,
        "judgement": None,
        "hashtags": None
    }

def save_caption_result(result: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def caption_images():
    config = load_config()
    model_name = config["captioning"]["blip_model"]
    image_folder = config["captioning"]["image_folder"]
    output_folder = config["captioning"]["output_folder"]
    prompt = config["captioning"]["prompt"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_blip_model(model_name)

    print(f"[Loading Model] {model_name}")
    print(f"[Image Folder] {image_folder}")
    print(f"[Output Folder] {output_folder}")
    print(f"[Prompt] {prompt}")

    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(image_folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_folder, json_filename)

        if os.path.exists(json_path):
            print(f"[Already Exists] {json_path}")
            continue

        image_path = os.path.join(image_folder, filename)
        print(f"[Processing] {filename}")

        print(f"[Image Path] {image_path}")
        print(f"[Output Path] {json_path}")

        try:
            caption = generate_caption(image_path, processor, model, prompt, device)
            result = create_caption_result(filename, image_path, caption)
            save_caption_result(result, json_path)
            print(f"✅ Saved: {json_path}")
        except Exception as e:
            print(f"❌ Error: {filename} - {e}")
