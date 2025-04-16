import os
import json
import torch
from utils.config_loader import load_config
from utils.judgement_utils import load_filtering_model, filter_caption
from datetime import datetime
import shutil
from pathlib import Path

DATA_DIR = Path("./data")
PASS_DIR = DATA_DIR / "images_filtered" / "step-4" / "적합"
FAIL_DIR = DATA_DIR / "images_filtered" / "step-4" / "부적합"

def process_json_files(input_folder: str, output_folder: str, tokenizer, model, device: str, log_path: str, config):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_entries = []
    json_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".json")]

    for json_file in json_files:
        json_path = os.path.join(input_folder, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_path = data['filepath']
        judgement = data.get("judgement")

        if config.get("skip_if_judged", True) and data.get("judgement") in ["Suitable", "Unsuitable"]:
            copy_image_by_judgement(img_path, judgement)
            continue

        caption = data.get("caption", "").strip()

        if not caption:
            print(f"[⚠️] 캡션 없음 → {json_file}")

        if caption:
            judgement, filter_response = filter_caption(caption, tokenizer, model, device, config)
        else:
            judgement, filter_response = "No Caption", "No caption text provided."

        data["judgement"] = judgement
        data[config.get("response_field_name", "filter_response")] = filter_response

        output_path = os.path.join(output_folder, json_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[✓] Judged {json_file} → {judgement} | Keywords Found: {config.get('suitable_keywords')}")
        log_entries.append({"file": json_file, "judgement": judgement, "response": filter_response})

        copy_image_by_judgement(img_path, judgement)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_path, f"judgement_log_{timestamp}.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, ensure_ascii=False, indent=2)

def copy_image_by_judgement(img_path, judgement):
    if not img_path or not os.path.exists(img_path):
        print(f"[이미지 없음] {img_path}")
        return

    file_name = os.path.basename(img_path)

    if judgement:
        target_dir = PASS_DIR
    else:
        target_dir = FAIL_DIR

    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(img_path, os.path.join(target_dir, file_name))
    print(f"[이미지 복사됨] {img_path} → {target_dir}")

if __name__ == "__main__":
    config = load_config()
    input_folder = config["judgement"]["input_folder"]
    output_folder = config["judgement"]["output_folder"]
    model_id = config["judgement"]["model"]
    log_path = config["judgement"].get("log_path")

    config = config['judgement']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_filtering_model(model_id)
    process_json_files(input_folder, output_folder, tokenizer, model, device, log_path, config)