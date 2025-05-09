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
    device: torch.device,
    torch_dtype=torch.float16,
    device_map="auto"
) -> Tuple[AutoProcessor, AutoModelForVision2Seq]:
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    model.to(device)
    model.eval()
    return processor, model

def generate_caption(
    image_path: str,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    prompt: str,
    device: torch.device
) -> Tuple[str, str]:
    img = Image.open(image_path).convert("RGB")
    formatted = f"<|system|>You are a helpful assistant.<|user|><image>{prompt}<|assistant|>"
    inputs = processor(images=img, text=formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
        )
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text, text

def process_captioning(
    json_dir: str,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
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

        raw, cap = "", ""
        for attempt in range(2):
            logger.info(f"[캡션 생성] {fname} (시도 {attempt+1})")
            try:
                raw, cap = generate_caption(img_path, processor, model, prompt, device)
            except Exception as e:
                logger.error(f"[캡션 실패] {fname} 시도 {attempt+1}: {e}", exc_info=True)
                torch.cuda.empty_cache()
                continue

            if cap:
                break
            logger.warning(f"[빈 캡션] {fname} 시도 {attempt+1} → 재시도")
            torch.cuda.empty_cache()

        rec.update({"raw_caption": raw, "caption": cap})
        save_result(rec, jp)
        logger.debug(f"[저장 완료] {jp}")
