import os
import json
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from utils.constants import VALID_EXTENSIONS
from utils.file_io import save_result

TOP_K = 3  # Top‑K 평균에 사용할 K 값


@lru_cache(maxsize=2)
def load_clip_model(name: str) -> Tuple[CLIPProcessor, CLIPModel, torch.device]:
    """CLIP 모델 + 프로세서 로드 (캐시)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = CLIPProcessor.from_pretrained(name)
    mdl = CLIPModel.from_pretrained(name).to(device).eval()
    return proc, mdl, device


def compute_similarity(
    img: Image.Image,
    texts: List[str],
    proc: CLIPProcessor,
    mdl: CLIPModel,
    device: torch.device,
) -> np.ndarray:
    """이미지와 텍스트 리스트 간 유사도 배열 반환"""
    inputs = proc(text=texts, images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_emb = mdl.get_image_features(pixel_values=inputs.pixel_values)
        txt_emb = mdl.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return (img_emb @ txt_emb.T).squeeze(0).cpu().numpy()


def calc_topk_metrics(scores: np.ndarray, prompts: List[str]) -> Tuple[float, float, str]:
    """최대값, Top‑K 평균, 최고점 프롬프트 반환"""
    idx_sorted = np.argsort(scores)[::-1]  # 내림차순
    max_idx = idx_sorted[0]
    topk_idx = idx_sorted[:TOP_K]
    return float(scores[max_idx]), float(scores[topk_idx].mean()), prompts[max_idx]


def classify_scene_object(
    img_path: str,
    prompts: Dict[str, List[str]],
    proc: CLIPProcessor,
    mdl: CLIPModel,
    device: torch.device,
) -> Dict[str, float]:
    """이미지‑scene / object 프롬프트 점수 딕셔너리 반환"""
    img = Image.open(img_path).convert("RGB")

    scene_scores = compute_similarity(img, prompts["scene"], proc, mdl, device)
    object_scores = compute_similarity(img, prompts["object"], proc, mdl, device)

    scene_max, scene_topk, top_scene_prompt = calc_topk_metrics(scene_scores, prompts["scene"])
    object_max, object_topk, top_object_prompt = calc_topk_metrics(object_scores, prompts["object"])

    return {
        "scene_max": scene_max,
        "scene_topk_avg": scene_topk,
        "object_max": object_max,
        "object_topk_avg": object_topk,
        "gap_max": scene_max - object_max,
        "gap_avg": scene_topk - object_topk,
        "top_scene_prompt": top_scene_prompt,
        "top_object_prompt": top_object_prompt,
    }


def process_clip_filtering(
    json_dir: str,
    data_dir: str,
    model_name: str,
    prompts: Dict[str, List[str]],
):
    """이미지 디렉토리를 돌며 CLIP score 저장"""
    os.makedirs(json_dir, exist_ok=True)
    proc, mdl, device = load_clip_model(model_name)

    for img_name in os.listdir(data_dir):
        if not img_name.lower().endswith(VALID_EXTENSIONS):
            continue
        img_path = os.path.join(data_dir, img_name)
        if not os.path.isfile(img_path):
            continue

        fname, _ = os.path.splitext(img_name)
        json_path = os.path.join(json_dir, f"{fname}.json")

        # JSON 로드 또는 초기화
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as jf:
                rec = json.load(jf)
        else:
            rec = {"filename": fname, "filepath": img_path}

        # CLIP score 계산
        metrics = classify_scene_object(img_path, prompts, proc, mdl, device)
        rec.update(metrics)
        save_result(rec, json_path)
        print("[Saved]", json_path)