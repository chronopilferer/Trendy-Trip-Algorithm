import logging
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from utils.constants import VALID_EXTENSIONS
from utils.file_io import save_result
from utils.io import load_record

logger = logging.getLogger(__name__)
TOP_K = 3  

@lru_cache(maxsize=2)
def load_clip_model(name: str) -> Tuple[CLIPProcessor, CLIPModel, torch.device]:
    """
    CLIP 프로세서와 모델을 로드하여 반환합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(name)
    model = CLIPModel.from_pretrained(name).to(device).eval()
    return processor, model, device

def compute_similarity(
    img: Image.Image,
    texts: List[str],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
) -> np.ndarray:
    """
    이미지와 텍스트 리스트 간 유사도 배열을 반환합니다.
    """
    inputs = processor(text=texts, images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_emb = model.get_image_features(pixel_values=inputs.pixel_values)
        txt_emb = model.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return (img_emb @ txt_emb.T).squeeze(0).cpu().numpy()

def calc_topk_metrics(scores: np.ndarray, prompts: List[str]) -> Tuple[float, float, str]:
    """
    최대값, Top-K 평균, 최고점 프롬프트를 반환합니다.
    """
    idx_sorted = np.argsort(scores)[::-1]
    max_idx = idx_sorted[0]
    topk_idx = idx_sorted[:TOP_K]
    return float(scores[max_idx]), float(scores[topk_idx].mean()), prompts[max_idx]

def classify_scene_object(
    img_path: Path,
    prompts: Dict[str, List[str]],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
) -> Dict[str, float]:
    """
    scene/object 프롬프트 점수를 계산해 딕셔너리로 반환합니다.
    """
    img = Image.open(str(img_path)).convert("RGB")
    scene_scores = compute_similarity(img, prompts["scene"], processor, model, device)
    object_scores = compute_similarity(img, prompts["object"], processor, model, device)

    scene_max, scene_topk, _ = calc_topk_metrics(scene_scores, prompts["scene"])
    object_max, object_topk, _ = calc_topk_metrics(object_scores, prompts["object"])

    return {
        "scene_max": scene_max,
        "scene_topk_avg": scene_topk,
        "object_max": object_max,
        "object_topk_avg": object_topk,
        "gap_max": scene_max - object_max,
        "gap_avg": scene_topk - object_topk,
    }

def process_clip_filtering(
    json_dir: Path,
    data_dir: Path,
    model_name: str,
    prompts: Dict[str, List[str]],
) -> None:
    """
    이미지 디렉토리를 순회하며 CLIP 메트릭을 JSON에 저장합니다.
    실패 시 로깅 후 건너뜁니다.
    """
    json_dir.mkdir(parents=True, exist_ok=True)
    processor, model, device = load_clip_model(model_name)

    for img_path in data_dir.iterdir():
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        try:
            json_path = json_dir / f"{img_path.stem}.json"
            rec = load_record(json_path, defaults={"file_path": str(img_path)})

            metrics = classify_scene_object(img_path, prompts, processor, model, device)
            rec.update(metrics)
            save_result(rec, str(json_path))
            logger.info(f"Saved CLIP metrics for {img_path.name}")
        except Exception as e:
            logger.error(f"Failed CLIP processing {img_path.name}: {e}", exc_info=True)