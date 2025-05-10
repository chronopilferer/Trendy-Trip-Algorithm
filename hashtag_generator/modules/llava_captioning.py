import os
import json
import logging
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig
)
from typing import Tuple
from utils.file_io import save_result

logger = logging.getLogger(__name__)

def load_img_to_text_model(
    model_name: str,
    device: torch.device
) -> Tuple[AutoProcessor, LlavaForConditionalGeneration]:
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model

def generate_caption(
    image_path: str,
    processor: AutoProcessor,
    model: LlavaForConditionalGeneration,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64
) -> str:
    img = Image.open(image_path).convert("RGB")

    formatted_prompt = f"USER: <image>\n{prompt} ASSISTANT:"

    inputs = processor(formatted_prompt, img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            use_cache=True,
        )

    text = processor.decode(outputs[0], skip_special_tokens=True)
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    return text

def process_captioning(
    json_dir: str,
    processor: AutoProcessor,
    model: LlavaForConditionalGeneration,
    device: torch.device
):
    for fname in os.listdir(json_dir):
        if not fname.lower().endswith(".json"):
            continue

        path = os.path.join(json_dir, fname)
        try:
            rec = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            logger.error(f"[JSON 로드 실패] {path}: {e}")
            continue

        img_path = rec.get("file_path", "")
        if not img_path or not os.path.isfile(img_path):
            logger.warning(f"[이미지 없음] {path}")
            continue

        caption = ""
        for attempt in range(2):
            logger.info(f"[캡션 생성] {fname} (시도 {attempt+1})")
            try:
                caption = generate_caption(
                    img_path, processor, model,
                    rec.get("prompt", "Describe clearly what you see."),
                    device
                )
            except Exception as e:
                logger.error(f"[캡션 실패] {fname} 시도 {attempt+1}: {e}", exc_info=True)
                torch.cuda.empty_cache()
                continue

            if caption:
                break
            logger.warning(f"[빈 캡션] {fname} 시도 {attempt+1} → 재시도")
            torch.cuda.empty_cache()

        rec["raw_caption"] = rec.get("raw_caption", "")
        rec["caption"] = caption
        save_result(rec, path)
        logger.debug(f"[저장 완료] {path}")