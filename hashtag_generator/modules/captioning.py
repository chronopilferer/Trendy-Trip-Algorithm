import os
import json
import logging
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Tuple

from utils.file_io import save_result

logger = logging.getLogger(__name__)

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
) -> Tuple[str, str]:
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
        raw_text = raw_text.split("<|assistant|>")[-1].strip()
    return raw_text, raw_text

def process_captioning(
    json_dir: str,
    processor,
    model,
    prompt: str,
    device: torch.device
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

        img_path = rec.get("file_path")
        if not img_path or not os.path.isfile(img_path):
            logger.warning(f"[이미지 없음] {json_path}")
            continue

        raw_caption, caption = "", ""
        for attempt in range(2):
            logger.info(f"[캡션 생성 중] {fname} (시도 {attempt+1})")
            try:
                with torch.no_grad():
                    raw_caption, caption = generate_caption(
                        img_path, processor, model, prompt, device
                    )
            except Exception as e:
                logger.error(f"[캡션 생성 실패] {fname} 시도 {attempt+1}: {e}", exc_info=True)
                torch.cuda.empty_cache()
                continue

            if raw_caption.strip() and caption.strip():
                break
            else:
                logger.warning(f"[빈 캡션] {fname} 시도 {attempt+1}에서 캡션이 비어있음, 재시도합니다.")
                torch.cuda.empty_cache()

        rec.update({
            "caption": caption,
            "raw_caption": raw_caption
        })
        save_result(rec, json_path)
        logger.debug(f"[저장 완료] {json_path}")
