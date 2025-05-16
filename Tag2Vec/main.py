import logging
import json
import numpy as np
import torch
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
from constants import MODEL_CONFIG

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def load_models(model_config, aliases, device):
    models = {}
    for alias, model_id in model_config.items():
        if aliases and alias not in aliases:
            continue
        logging.info(f"Loading model {alias} ({model_id}) on {device}")
        models[alias] = SentenceTransformer(model_id, device=device)
    return models

def embed_hashtags(hashtags, models, batch_size, device):
    embeddings = {}
    eps = 1e-8
    for alias, model in models.items():
        emb = model.encode(
            hashtags,
            batch_size=batch_size,
            show_progress_bar=False,
            device=device,
            convert_to_numpy=True
        )
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        normalized = emb / (norms + eps)
        embeddings[alias] = normalized.astype(float).tolist()
    return embeddings

def process_file(json_path: Path, models, batch_size, device):
    rec = json.loads(json_path.read_text(encoding="utf-8"))
    if rec.get("pass_type") != "pass":
        logging.info(f"▶ skip: {json_path.name} (pass_type={rec.get('pass_type')})")
        return

    hashtags = rec.get("hashtag", [])
    if not isinstance(hashtags, list) or not hashtags:
        logging.info(f"▶ skip: {json_path.name} (유효한 hashtag 없음)")
        return

    rec["hashtag_embeddings"] = embed_hashtags(hashtags, models, batch_size, device)
    json_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"✅ 업데이트 완료: {json_path.name}")

def run_pipeline(input_dir: str, batch_size: int = 16, device: str = None, model_aliases: list = None):
    setup_logging()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    models = load_models(MODEL_CONFIG, model_aliases, device)
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"입력 디렉토리 없음: {input_dir}")

    for json_file in input_path.glob("*.json"):
        process_file(json_file, models, batch_size, device)

if __name__ == "__main__":
    cfg_path = Path("configs/config.yml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} 파일을 찾을 수 없습니다.")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    run_pipeline(
        input_dir=cfg["input_dir"],
        batch_size=cfg.get("batch_size", 16),
        device=cfg.get("device"),
        model_aliases=cfg.get("model_aliases")
    )
