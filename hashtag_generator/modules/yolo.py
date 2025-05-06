import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple

from utils.constants import TARGET_LABELS, VALID_EXTENSIONS
from utils.file_io import save_result

def load_yolo_model(model_path: str) -> YOLO:
    """
    YOLO 모델을 로드하여 반환합니다.
    """
    return YOLO(model_path)

def detect_objects(
    model: YOLO,
    image: np.ndarray
) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    """
    이미지에서 객체를 검출하여 라벨과 바운딩 박스 리스트를 반환합니다.
    """
    results = model(image, verbose=False)[0]
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    labels = [model.names[cid] for cid in cls_ids]
    bboxes = [tuple(box) for box in results.boxes.xyxy.cpu().numpy().astype(int)]
    return labels, bboxes

def compute_area_ratios(
    labels: List[str],
    boxes: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int]
) -> dict:
    """
    전체 이미지 대비 사람과 음식 객체 면적 비율을 계산하여 반환합니다.
    """
    h, w = image_shape
    total_area = h * w if h and w else 1
    person_area = sum(
        (x2 - x1) * (y2 - y1)
        for lbl, (x1, y1, x2, y2) in zip(labels, boxes)
        if lbl == 'person'
    )
    food_area = sum(
        (x2 - x1) * (y2 - y1)
        for lbl, (x1, y1, x2, y2) in zip(labels, boxes)
        if lbl in TARGET_LABELS and lbl != 'person'
    )
    return {
        'person_area_ratio': float(person_area / total_area),
        'food_area_ratio': float(food_area / total_area)
    }

def process_yolo_filtering(json_dir: str, data_dir: str, model_path: str) -> None:
    """
    이미지 디렉토리를 순회하며 YOLO 객체 검출 후 면적 비율을 JSON에 누적 저장합니다.
    """
    os.makedirs(json_dir, exist_ok=True)
    model = load_yolo_model(model_path)

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

        # 객체 검출 및 면적 비율 계산
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'Cannot load image: {img_path}')
        labels, boxes = detect_objects(model, img)
        ratios = compute_area_ratios(labels, boxes, img.shape[:2])

        rec.update(ratios)
        save_result(rec, json_path)
        print(f'[Saved] {json_path}')