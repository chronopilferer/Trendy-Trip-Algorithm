import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from modules.captioning import load_img_to_text_model

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
        modified_prompt = prompt.strip() + "\nAnswer:"
        inputs = processor(images=image, text=modified_prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=256)
        caption = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if caption.startswith(modified_prompt):
            caption = caption[len(modified_prompt):].strip()

        results[f"Prompt_{idx+1}"] = {
            "prompt": prompt,
            "caption": caption
        }

        print(f"[Prompt {idx+1}] {prompt}")
        print(f"[Caption {idx+1}] {caption}\n")

    return results
import pprint

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_img_to_text_model(model_name="Salesforce/instructblip-vicuna-7b")

    prompts = [
        "Describe everything you clearly see in the image, using natural, complete sentences. Focus on visible objects, colors, textures, and actions. Do not add information that is not visibly present.",
        "List the objects, materials, and visible actions you clearly see in the image, using natural, complete sentences. Focus only on what is visually present without any assumptions about context, purpose, or unseen elements.",
        "Factually describe only the objects, materials, and visible actions clearly present in the image. Use complete, natural sentences. Do not describe emotions, atmosphere, purposes, or any details that cannot be directly observed."
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