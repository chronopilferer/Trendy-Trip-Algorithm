import os
import json
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from utils.file_io import save_result

def load_img_to_text_model(model_name: str, torch_dtype=torch.float16, device_map: str = "auto"):
    print(f"[Loading Model] {model_name}")
    processor = InstructBlipProcessor.from_pretrained(model_name)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    return processor, model

def generate_caption(image_path: str, processor, model, prompt: str, device: torch.device) -> str:
    image = Image.open(image_path).convert("RGB")
    modified_prompt = prompt.strip() + "\nAnswer:"
    inputs = processor(images=image, text=modified_prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=256)
    caption = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    if caption.startswith(modified_prompt):
        caption = caption[len(modified_prompt):].strip()

    return caption

def process_captioning(
    json_dir: str,
    model_name: str,
    prompt: str
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_img_to_text_model(model_name=model_name)

    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]

    for fname in json_files:
        json_path = os.path.join(json_dir, fname)

        with open(json_path, "r", encoding="utf-8") as jf:
            rec = json.load(jf)

        if rec.get("pass") is False:
            continue

        image_path = rec.get("filepath")
        if not image_path or not os.path.isfile(image_path):
            print(f"[이미지 없음] {json_path}")
            continue

        print(f"[캡션 생성 시작] {fname}")
        print(f"[이미지 경로] {image_path}")

        try:
            caption = generate_caption(image_path, processor, model, prompt, device)
            rec.update({"caption": caption})
            print(f"caption: {caption}")
            save_result(rec, json_path)
            print(f"✅ 캡션 저장 완료: {json_path}")

        except Exception as e:
            print(f"❌ 에러 발생: {fname} - {e}")
