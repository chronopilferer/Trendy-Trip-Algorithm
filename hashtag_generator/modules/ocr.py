import os
import json
import cv2
import numpy as np
import easyocr
from typing import List, Tuple

from utils.constants import VALID_EXTENSIONS
from utils.file_io import save_result

def load_ocr_model(langs: List[str] = ['ko','en'], gpu: bool = True) -> easyocr.Reader:
    """
    EasyOCR 리더를 초기화하여 반환합니다.
    """
    return easyocr.Reader(langs, gpu=gpu)

def detect_text_boxes(
    model: easyocr.Reader,
    image: np.ndarray
) -> List[List[Tuple[float, float]]]:
    """
    이미지에서 검출된 텍스트 박스 좌표 리스트를 반환합니다.
    """
    results = model.readtext(image)
    boxes: List[List[Tuple[float, float]]] = []
    for coords, _, _ in results:
        if len(coords) == 4:
            boxes.append(coords)
        else:
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            boxes.append([
                (min(xs), min(ys)),
                (max(xs), min(ys)),
                (max(xs), max(ys)),
                (min(xs), max(ys))
            ])
    return boxes

def compute_text_area_ratio(
    boxes: List[List[Tuple[float, float]]],
    image_shape: Tuple[int, int]
) -> float:
    """
    텍스트 박스 면적 합계를 이미지 전체 면적으로 나눈 비율을 반환합니다.
    """
    h, w = image_shape[:2]
    total_area = h * w
    if total_area == 0:
        return 0.0
    text_area = sum(
        cv2.contourArea(np.array([[int(x), int(y)] for x, y in box], np.int32))
        for box in boxes
    )
    return float(text_area / total_area)

def process_single_ocr(
    fpath: str,
    reader: easyocr.Reader
) -> dict:
    """
    단일 이미지 파일에 대해 OCR을 수행하고, 텍스트 면적 비율과 박스 개수를 반환합니다.
    """
    image = cv2.imread(fpath)
    if image is None:
        raise FileNotFoundError(f'Cannot load image: {fpath}')

    boxes = detect_text_boxes(reader, image)
    ratio = compute_text_area_ratio(boxes, image.shape)
    num_boxes = len(boxes)

    return {
        'text_area_ratio': ratio,
        'num_text_boxes': num_boxes
    }

def process_ocr_filtering(
    json_dir: str,
    data_dir: str
) -> None:
    """
    데이터 디렉토리의 이미지들을 순회하며 OCR 메트릭을 JSON 파일에 누적 저장합니다.
    """
    os.makedirs(json_dir, exist_ok=True)
    reader = load_ocr_model()

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
            rec = { 'filename': fname, 'filepath': img_path }

        # OCR 메트릭 계산 및 저장
        metrics = process_single_ocr(img_path, reader)
        rec.update(metrics)
        save_result(rec, json_path)
        print(f'[Saved] {json_path}')