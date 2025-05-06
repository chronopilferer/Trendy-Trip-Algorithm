import os
import json
import cv2
import numpy as np
from utils.constants import (
    MIN_WIDTH,
    MIN_HEIGHT,
    DARKNESS_THRESHOLD,
    BRIGHTNESS_THRESHOLD,
    ENTROPY_THRESHOLD,
    VALID_EXTENSIONS
)
from utils.file_io import save_result

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
) -> dict:
    """
    이미지의 너비, 높이와 기준 대비 해상도 비율을 계산하여 반환합니다.
    """
    h, w = image.shape[:2]
    resolution_ratio = min(w / min_width, h / min_height)
    return {
        'image_width': int(w),
        'image_height': int(h),
        'resolution_ratio': float(resolution_ratio)
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

def process_single_image_filtering(fpath: str) -> dict:
    """
    단일 이미지에 대해 밝기, 해상도, 엔트로피 점수를 계산하고 레코드로 반환합니다.
    """
    image = cv2.imread(fpath)
    if image is None:
        raise FileNotFoundError(f'Cannot load image: {fpath}')

    # 각종 메트릭 계산
    brightness = compute_brightness(image)
    resolution = compute_resolution_metrics(image)
    entropy = compute_entropy(image)

    # 기준 대비 통과 여부 판단
    brightness_ok = (brightness >= DARKNESS_THRESHOLD) and (brightness <= BRIGHTNESS_THRESHOLD)
    resolution_ok = (resolution['image_width'] >= MIN_WIDTH) and (resolution['image_height'] >= MIN_HEIGHT)
    entropy_ok = (entropy >= ENTROPY_THRESHOLD)
    passed = brightness_ok and resolution_ok and entropy_ok

    # 결과 반환
    return {
        'brightness_score': brightness,
        **resolution,
        'entropy_score': entropy,
        'step1_pass': bool(passed)
    }

def process_img_filtering(json_dir: str, data_dir: str) -> None:
    """
    주어진 이미지 디렉토리를 순회하며 각 이미지별 JSON을 생성/업데이트합니다.
    모든 메트릭을 하나의 JSON 파일에 누적 저장하며, 별도 복사나 분기는 없습니다.
    """
    os.makedirs(json_dir, exist_ok=True)

    for img_name in os.listdir(data_dir):
        if not img_name.lower().endswith(VALID_EXTENSIONS):
            continue
        img_path = os.path.join(data_dir, img_name)
        if not os.path.isfile(img_path):
            continue

        fname, _ = os.path.splitext(img_name)
        json_path = os.path.join(json_dir, f'{fname}.json')

        # 기존 JSON 로드 또는 초기화
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as jf:
                rec = json.load(jf)
        else:
            rec = {'filename': fname, 'filepath': img_path}

        # 필터링 수행 및 JSON 업데이트
        metrics = process_single_image_filtering(img_path)
        rec.update(metrics)
        save_result(rec, json_path)
        print(f'[Saved] {json_path}')
