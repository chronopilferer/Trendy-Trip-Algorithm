import os
import json
import torch
from PIL import Image
from transformers import (
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
)

def load_model():
    print("[Loading Model] InstructBLIP-Vicuna7B")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

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

def caption_images(image_folder, output_folder, processor, model, prompt, device):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_folder, json_filename)

        if os.path.exists(json_path):
            print(json_path)
            continue

        image_path = os.path.join(image_folder, filename)
        print(f"[Processing with InstructBLIP-Vicuna7B] {filename}")

        try:
            caption = generate_caption(image_path, processor, model, prompt, device)
            result = {
                "filename": filename,
                "filepath": image_path,
                "caption": caption,
                "judgement": None,     
                "hashtags": None       
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"✅ Saved: {json_path}")

        except Exception as e:
            print(f"❌ Error occurred: {filename} - {e}")

if __name__ == "__main__":
    image_folder = "./images"
    output_folder = "./captions"
    prompt = (
        "Describe the image in meticulous detail based solely on what can be visually observed. "
        "Include comprehensive and precise descriptions of the setting, objects, people, colors, textures, facial expressions, posture, and any visible actions or interactions. "
        "Avoid guessing or making assumptions beyond what is clearly shown in the image. Use complete sentences and maintain a focus on factual accuracy and visual clarity."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model()
    caption_images(image_folder, output_folder, processor, model, prompt, device)
