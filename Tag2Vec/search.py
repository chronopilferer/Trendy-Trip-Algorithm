import json
import yaml
import numpy as np
import torch
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from constants import MODEL_CONFIG

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def load_config(config_path: str = "config.yml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_embeddings(
    input_dir: str,
    model_alias: str
) -> List[Tuple[str, List[str], np.ndarray]]:
    data = []
    for fn in Path(input_dir).glob("*.json"):
        rec = json.loads(fn.read_text(encoding="utf-8"))
        hashtags = rec.get("hashtag", [])
        vecs = rec.get("hashtag_embeddings", {}).get(model_alias)
        if not vecs:
            logging.warning(f"{fn.name}에 모델 '{model_alias}' 임베딩 없음. 건너뜀.")
            continue
        arr = np.array(vecs, dtype=np.float32)
        if hashtags:
            data.append((fn.name, hashtags, arr))
    logging.info(f"Loaded {len(data)} records for 모델='{model_alias}'")
    return data

def search_by_vector(
    query_vec: np.ndarray,
    data: List[Tuple[str, List[str], np.ndarray]],
    top_k: int = 10
) -> List[Dict]:
    results = []
    for fname, tags, arr in data:
        scores = arr @ query_vec
        idx = int(np.argmax(scores))
        results.append({
            "file": fname,
            "top_hashtag": tags[idx],
            "similarity": float(scores[idx])
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def load_text_model(alias: str, device: str) -> SentenceTransformer:
    model_id = MODEL_CONFIG[alias]
    logging.info(f"Loading text model {alias} on {device}")
    return SentenceTransformer(model_id, device=device)

def encode_text(
    text: str,
    model: SentenceTransformer
) -> np.ndarray:
    vec = model.encode([text], convert_to_numpy=True, device=model.device)
    norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    return (vec / norm)[0]

def search_by_text(
    query_text: str,
    model_alias: str,
    data: List[Tuple[str, List[str], np.ndarray]],
    top_k: int = 10
) -> List[Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_model = load_text_model(model_alias, device)
    q_vec = encode_text(query_text, text_model)
    return search_by_vector(q_vec, data, top_k)

def save_results(
    query_text: str,
    model_alias: str,
    results: List[Dict],
    output_root: str = "output/results"
):
    safe_query = query_text.replace("/", "_").replace(" ", "_")
    output_dir = Path(output_root) / safe_query
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_alias}_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved results to {output_path}")

if __name__ == "__main__":
    setup_logging()
    config = load_config("configs/config.yml")

    INPUT_DIR = config["input_dir"]
    MODEL_ALIASES = config["model_aliases"]
    TOP_K = config.get("top_k", 5)
    QUERY_TEXT = config.get("query_text", "하늘")

    for model_alias in MODEL_ALIASES:
        logging.info(f"===== 모델: {model_alias} =====")
        dataset = load_embeddings(INPUT_DIR, model_alias)
        if not dataset:
            logging.warning(f"{model_alias}에 대한 데이터셋이 비어 있음. 스킵.")
            continue

        topt = search_by_text(QUERY_TEXT, model_alias, dataset, top_k=TOP_K)
        logging.info(f"=== 텍스트 검색('{QUERY_TEXT}') Top {TOP_K} (모델: {model_alias}) ===")
        for r in topt:
            logging.info(r)

        save_results(QUERY_TEXT, model_alias, topt)
