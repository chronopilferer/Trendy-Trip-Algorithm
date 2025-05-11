import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import os
import shutil

logger = logging.getLogger(__name__)

def load_record(json_path: Path, defaults: Dict[str, Any]) -> Dict[str, Any]:
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {json_path.name}: {e}")
    return defaults.copy()

def load_json_records(json_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    json_dir = Path(json_dir)

    if not json_dir.exists() or not json_dir.is_dir():
        logger.error(f"Invalid JSON directory: {json_dir}")
        return records

    json_paths = list(json_dir.glob("*.json"))
    if not json_paths:
        logger.warning(f"No JSON files found in: {json_dir}")
        return records

    total = len(json_paths)
    for idx, path in enumerate(json_paths, start=1):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {path.name}: {e}")
        if idx % 100 == 0 or idx == total:
            logger.info(f"Loaded {idx}/{total} JSON files")

    return records

def save_result(result: dict, output_path: str):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[파일 저장 실패] {output_path}: {e}")

def copy_image(img_path: str, filtered_base_dir: str, judgement: str):

    if not img_path or not os.path.exists(img_path):
        print(f"[이미지 없음] {img_path}")
        return
    
    filename = os.path.basename(img_path)
    target_dir = os.path.join(filtered_base_dir, judgement)
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(img_path, os.path.join(target_dir, filename))
    print(f"[이미지 복사됨] {img_path} → {target_dir}")

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)