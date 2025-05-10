import os
import json
import logging
import time
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from typing import Tuple, Dict, Any, Union
from utils.file_io import save_result

logger = logging.getLogger(__name__)
torch.manual_seed(42)

def load_img_to_text_model(
    model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto"
) -> Tuple[AutoProcessor, AutoModelForVision2Seq]:
    gpu_available = torch.cuda.is_available()
    quant_config = (
        BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4"
        )
        if gpu_available else None
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=not gpu_available,
        trust_remote_code=True
    )

    if hasattr(torch, "compile"):
        model = torch.compile(model)
    model.eval()
    return processor, model

def is_valid_caption(
    caption: str,
    prompt: str,
    min_words: int = 3
) -> bool:
    if not caption:
        return False
    text = caption.strip()
    if text == prompt.strip():
        return False
    if len(text.split()) < min_words:
        return False
    return True

def generate_caption(
    image: Union[str, Image.Image],
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 128,
    gen_kwargs: Dict[str, Any] = None
) -> str:
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image

    raw_inputs = processor(images=img, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in raw_inputs.items()}

    gen_args = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "num_beams": 1,
        "do_sample": False,
    }
    if gen_kwargs:
        gen_args.update(gen_kwargs)

    with torch.inference_mode():
        outputs = model.generate(**gen_args)
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return caption

def process_captioning(
    json_dir: str,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 128
):
    images_cache: Dict[str, Image.Image] = {}

    for fname in os.listdir(json_dir):
        if not fname.lower().endswith(".json"):
            continue

        path = os.path.join(json_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            logger.error(f"[JSON load failed] {path}: {e}")
            continue

        img_path = rec.get("file_path")
        if not img_path or not os.path.isfile(img_path):
            logger.warning(f"[Image not found] {path}")
            continue

        if img_path not in images_cache:
            images_cache[img_path] = Image.open(img_path).convert("RGB")
        img = images_cache[img_path]

        start_time = time.time()
        try:
            cap = generate_caption(
                img,
                processor,
                model,
                prompt,
                device,
                max_new_tokens=max_new_tokens
            )
        except Exception as e:
            logger.error(f"[Generation error] {fname}: {e}", exc_info=True)
            torch.cuda.empty_cache()
            cap = ""

        if is_valid_caption(cap, prompt):
            caption = cap
        else:
            caption = "I cannot describe this image clearly."

        elapsed = time.time() - start_time
        logger.info(f"[{fname}] 처리 시간: {elapsed:.2f}s")

        rec["caption_instructblip"] = caption
        save_result(rec, path)
        logger.debug(f"[Saved] {path}")
