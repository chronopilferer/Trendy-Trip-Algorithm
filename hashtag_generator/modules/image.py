import logging
from pathlib import Path
import cv2
import numpy as np
from typing import Dict

from utils.constants import MIN_WIDTH, MIN_HEIGHT, VALID_EXTENSIONS
from utils.file_io import save_result
from utils.io import load_record

logger = logging.getLogger(__name__)

def compute_brightness(image: np.ndarray) -> float:
    """
    이미지의 HSV V 채널 평균값을 계산하여 밝기 점수로 반환합니다.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())

def compute_resolution_metrics(
    image: np.ndarray,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT
) -> Dict[str, float]:
    """
    이미지의 너비, 높이와 기준 대비 해상도 비율을 계산하여 반환합니다.
    """
    h, w = image.shape[:2]
    ratio = min(w / min_width, h / min_height)
    return {
        'image_width': float(w),
        'image_height': float(h),
        'resolution_ratio': float(ratio)
    }

def compute_entropy(image: np.ndarray) -> float:
    """
    그레이스케일 히스토그램 기반 엔트로피를 계산하여 반환합니다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    entropy_value = -np.sum([p * np.log2(p) for p in hist_norm if p > 0])
    return float(entropy_value)

def process_single_image(fpath: Path) -> Dict[str, float]:
    """
    단일 이미지 파일에 대해 여러 메트릭을 계산하여 반환합니다.
    """
    image = cv2.imread(str(fpath))
    if image is None:
        raise IOError(f"이미지 로드 실패: {fpath}")

    return {
        'brightness_score': compute_brightness(image),
        **compute_resolution_metrics(image),
        'entropy_score': compute_entropy(image),
    }

def process_img_filtering(json_dir: Path, data_dir: Path) -> None:
    """
    이미지 디렉토리를 순회하며 JSON 파일을 생성/업데이트합니다.
    각 이미지에 대해 메트릭을 계산해 누적 저장하고, 실패 시 로깅 후 건너뜁니다.
    """
    json_dir.mkdir(parents=True, exist_ok=True)

    for img_path in data_dir.iterdir():
        if not img_path.suffix.lower() in VALID_EXTENSIONS:
            continue
        try:
            json_path = json_dir / f"{img_path.stem}.json"
            defaults = {'file_path': str(img_path)}
            rec = load_record(json_path, defaults=defaults)

            metrics = process_single_image(img_path)
            rec.update(metrics)

            save_result(rec, str(json_path))
            logger.info(f"Saved metrics for {img_path.name}")

        except Exception as e:
            logger.error(f"Failed processing {img_path.name}: {e}", exc_info=True)
