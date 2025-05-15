import logging
from pathlib import Path
import cv2
import numpy as np
from typing import Dict

from img2hastag.utils.constants import MIN_WIDTH, MIN_HEIGHT, VALID_EXTENSIONS
from img2hastag.utils.io import save_result, load_record

logger = logging.getLogger(__name__)

def compute_brightness(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())

def compute_resolution_metrics(
    image: np.ndarray,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT
) -> Dict[str, float]:
    h, w = image.shape[:2]
    ratio = min(w / min_width, h / min_height)
    return {
        'image_width': float(w),
        'image_height': float(h),
        'resolution_ratio': float(ratio)
    }

def compute_entropy(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    entropy_value = -np.sum([p * np.log2(p) for p in hist_norm if p > 0])
    return float(entropy_value)

def process_single_image(fpath: Path) -> Dict[str, float]:
    image = cv2.imread(str(fpath))
    if image is None:
        raise IOError(f"이미지 로드 실패: {fpath}")

    return {
        'brightness_score': compute_brightness(image),
        **compute_resolution_metrics(image),
        'entropy_score': compute_entropy(image),
    }

def process_img_filtering(json_dir: Path, data_dir: Path, category: str) -> None:
    json_dir.mkdir(parents=True, exist_ok=True)

    for img_path in data_dir.iterdir():
        if not img_path.is_file() or not img_path.suffix.lower() in VALID_EXTENSIONS:
            continue
        try:
            json_path = json_dir / f"{img_path.stem}.json"

            rec = load_record(json_path, defaults={})

            if all(key in rec for key in [
                'brightness_score', 'entropy_score',
                'image_width', 'image_height', 'resolution_ratio'
            ]):
                logger.info(f"[스킵] 이미지 메트릭 존재함: {json_path.name}")
                continue

            rec.update({
                'file_path': str(img_path),
                'file_name': img_path.stem,
                'category': category
            })

            metrics = process_single_image(img_path)
            rec.update(metrics)

            save_result(rec, str(json_path))
            logger.info(f"Saved metrics for {img_path.name}")

        except Exception as e:
            logger.error(f"Failed processing {img_path.name}: {e}", exc_info=True)