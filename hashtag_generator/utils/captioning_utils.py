import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
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

def generate_caption(image_path, processor, model, prompt, device):
    image = Image.open(image_path).convert("RGB")
    modified_prompt = prompt.strip() + "\nAnswer:"
    inputs = processor(images=image, text=modified_prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    outputs = model.generate(**inputs, max_new_tokens=256)
    caption = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    if caption.startswith(modified_prompt):
        caption = caption[len(modified_prompt):].strip()

    return caption