import os
import json
import logging
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Tuple

from utils.file_io import save_result

logger = logging.getLogger(__name__)

def load_img_to_text_model(
    model_name: str,
    torch_dtype=torch.float16,
    device_map="auto",
    device: torch.device = None
):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    if device:
        model = model.to(device)
    model.eval()
    return processor, model

def generate_caption(
    image_path: str,
    processor,
    model,
    prompt: str,
    device: torch.device
) -> Tuple[str, str]:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
        )
    generated = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return generated, generated

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

        jp = os.path.join(json_dir, fname)
        try:
            rec = json.load(open(jp, encoding="utf-8"))
        except Exception as e:
            logger.error(f"[JSON 로드 실패] {jp}: {e}")
            continue

        img_path = rec.get("file_path")
        if not img_path or not os.path.isfile(img_path):
            logger.warning(f"[이미지 없음] {jp}")
            continue

        raw_caption, caption = "", ""
        for attempt in range(2):
            logger.info(f"[캡션 생성 중] {fname} (시도 {attempt+1})")
            try:
                raw_caption, caption = generate_caption(
                    img_path, processor, model, prompt, device
                )
            except Exception as e:
                logger.error(f"[캡션 실패] {fname} 시도 {attempt+1}: {e}", exc_info=True)
                torch.cuda.empty_cache()
                continue

            if caption:
                break
            logger.warning(f"[빈 캡션] {fname} 시도 {attempt+1}에서 캡션이 비어있음, 재시도합니다.")
            torch.cuda.empty_cache()

        rec.update({"caption": caption, "raw_caption": raw_caption})
        save_result(rec, jp)
        logger.debug(f"[저장 완료] {jp}")
