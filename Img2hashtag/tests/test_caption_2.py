import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import pprint

def load_llava_model(model_name: str, torch_dtype=torch.float16, device_map: str = "auto"):
    print(f"[Loading LLaVA Model] {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    return processor, model

def format_prompt(user_prompt: str) -> str:
    formatted = f"<|system|>You are a helpful assistant.<|user|><image>{user_prompt}<|assistant|>"
    return formatted

def test_multiple_prompts(
    image_path: str,
    processor,
    model,
    prompts: list,
    device: torch.device
) -> dict:
    """
    하나의 이미지에 대해 여러 프롬프트로 캡션을 생성하고 결과를 반환하는 함수
    """
    results = {}
    image = Image.open(image_path).convert("RGB")

    for idx, prompt in enumerate(prompts):
        formatted_prompt = format_prompt(prompt)
        
        inputs = processor(text=formatted_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256
        )

        caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        results[f"Prompt_{idx+1}"] = {
            "prompt": prompt,
            "caption": caption
        }

        print(f"[Prompt {idx+1}] {prompt}")
        print(f"[Caption {idx+1}] {caption}\n")

    return results

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor, model = load_llava_model(
        model_name="llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompts = [
        "List the objects, materials, and visible actions you clearly see in the image, using natural, complete sentences. Focus only on what is visually present without any assumptions about context, purpose, or unseen elements.",
        "Factually describe the objects, materials, and actions that are clearly visible in the image. Use precise and natural sentences.",
        "Write a factual description of the objects, materials, and visible actions in the image, avoiding any emotional or subjective language.",
        "In full sentences, describe only what is visually present in the image: objects, materials, and actions. Do not guess or assume."
    ]

    results = test_multiple_prompts(
        image_path="../data/images_raw/1158족욕카페_4.jpg",
        processor=processor,
        model=model,
        prompts=prompts,
        device=device
    )
    pprint.pprint(results)

    results = test_multiple_prompts(
        image_path="../data/images_raw/5L2F_5.jpg",
        processor=processor,
        model=model,
        prompts=prompts,
        device=device
    )
    pprint.pprint(results)
