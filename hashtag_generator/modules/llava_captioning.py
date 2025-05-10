import os
import json
import logging
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from typing import Tuple
from utils.file_io import save_result

logger = logging.getLogger(__name__)

def load_img_to_text_model(
    model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto"
) -> Tuple[AutoProcessor, AutoModelForVision2Seq]:
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config=quant_config,     
        torch_dtype=torch_dtype,              
        device_map=device_map,                
        trust_remote_code=True
    )

    # 4) 디바이스 할당 및 컴파일
    model.to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    model.eval()

    return processor, model

def generate_caption(
    image_path: str,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64
) -> str:
    img = Image.open(image_path).convert("RGB")
    formatted = (
        "<|system|>"
        "You are a helpful assistant. Describe only what you can clearly see."
        "<|user|>"
        "<image>"
        f"{prompt}"
        "<|assistant|>"
    )

    inputs = processor(images=img, text=formatted, return_tensors="pt").to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            early_stopping=True,
        )

    text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text

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

        path = os.path.join(json_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            logger.error(f"[JSON 로드 실패] {path}: {e}")
            continue

        img_path = rec.get("file_path")
        if not img_path or not os.path.isfile(img_path):
            logger.warning(f"[이미지 없음] {path}")
            continue

        new_cap = ""
        for attempt in range(2):
            logger.info(f"[캡션 생성] {fname} (시도 {attempt+1})")
            try:
                new_cap = generate_caption(img_path, processor, model, prompt, device)
            except Exception as e:
                logger.error(f"[캡션 실패] {fname} 시도 {attempt+1}: {e}", exc_info=True)
                torch.cuda.empty_cache()
                continue

            if new_cap:
                break
            logger.warning(f"[빈 캡션] {fname} 시도 {attempt+1} → 재시도")
            torch.cuda.empty_cache()

        rec["raw_caption"] = rec.get("raw_caption", "")
        rec["caption"]     = new_cap
        save_result(rec, path)
        logger.debug(f"[저장 완료] {path}")
