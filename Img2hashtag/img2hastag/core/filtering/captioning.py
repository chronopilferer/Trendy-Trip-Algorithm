import os
import json
import logging
import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BitsAndBytesConfig
from typing import Tuple

from Img2hashtag.img2hastag.utils.io import save_result

logger = logging.getLogger(__name__)

def load_img_to_text_model(model_name: str, device: torch.device) -> Tuple[InstructBlipProcessor, InstructBlipForConditionalGeneration]:
    try:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

        processor = InstructBlipProcessor.from_pretrained(model_name)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quant_config,  
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval() 
        logger.info(f"모델 {model_name} 로딩 완료.")
        return processor, model
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        raise e

def generate_caption(
    image_path: str,
    processor: InstructBlipProcessor,
    model: InstructBlipForConditionalGeneration,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 128
) -> str:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"[이미지 로드 실패] {image_path}: {e}")
        return ""

    formatted_prompt = f"{prompt.strip()}\nAnswer:"

    try:
        inputs = processor(images=img, text=formatted_prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        caption = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if caption.startswith(formatted_prompt):
            caption = caption[len(formatted_prompt):].strip()

        return caption
    except Exception as e:
        logger.error(f"[캡션 생성 실패] {image_path}: {e}")
        torch.cuda.empty_cache()
        return ""

def process_captioning(
    json_dir: str,
    processor: InstructBlipProcessor,
    model: InstructBlipForConditionalGeneration,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 128
):
    for fname in os.listdir(json_dir):
        if not fname.lower().endswith(".json"):
            continue

        json_path = os.path.join(json_dir, fname)
        try:
            with open(json_path, encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            logger.error(f"[JSON 로드 실패] {json_path}: {e}")
            continue

        img_path = rec.get("file_path", "")
        if not img_path or not os.path.isfile(img_path):
            logger.warning(f"[이미지 없음] {json_path}")
            continue

        caption = ""
        for attempt in range(2):
            logger.info(f"[캡션 생성] {fname} (시도 {attempt + 1})")
            try:
                caption = generate_caption(
                    img_path, processor, model,
                    rec.get("prompt", prompt),
                    device,
                    max_new_tokens=max_new_tokens
                )
            except Exception as e:
                logger.error(f"[캡션 실패] {fname} 시도 {attempt + 1}: {e}", exc_info=True)
                torch.cuda.empty_cache()
                continue

            if caption:
                break
            logger.warning(f"[빈 캡션] {fname} 시도 {attempt + 1} → 재시도")
            torch.cuda.empty_cache()

        rec["caption_instructblip"] = caption
        save_result(rec, json_path)
        logger.debug(f"[저장 완료] {json_path}")
