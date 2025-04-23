import os
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict
from ultralytics import YOLO

from utils.draw import draw_boxes
from utils.constants import DEFAULT_PERSON_AREA_THRESHOLD
from utils.file_io import save_result, copy_image

def load_yolo_model(model_path: str) -> YOLO:
    return YOLO(model_path)

def detect_objects(
    model: YOLO,
    image: np.ndarray
) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    results = model(image, verbose=False)[0]
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    labels = [model.names[cid] for cid in cls_ids]
    bboxes = [tuple(box) for box in results.boxes.xyxy.cpu().numpy().astype(int)]
    return labels, bboxes

def is_person_dominent(
    labels: List[str],
    boxes: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    threshold: float = DEFAULT_PERSON_AREA_THRESHOLD
) -> Tuple[bool, float]:
    h, w = image_shape
    if h == 0 or w == 0:
        return False, 0.0
    
    total_area = h * w
    person_area = sum(
        (x2 - x1) * (y2 - y1)
        for lbl, (x1, y1, x2, y2) in zip(labels, boxes)
        if lbl == 'person'
    )

    ratio = person_area / total_area
    return ratio > threshold, ratio

def process_single_image(
    model: YOLO,
    img_path: str,
    area_threshold: float,
    return_vis: bool
) -> Tuple[Dict, np.ndarray, np.ndarray]:

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'이미지 로딩 실패: {img_path}')

    labels, boxes = detect_objects(model, img)
    flag, ratio = is_person_dominent(labels, boxes, img.shape[:2], area_threshold)

    vis = draw_boxes(img.copy(), boxes) if return_vis else None

    result = {
        'person_dominant': bool(flag),
        'person_area_ratio': ratio,
        'pass': bool(not flag)
    }

    return result, vis, img

def process_yolo_filtering(
    json_dir: str,
    output_dir: str,
    model_path: str,
    vis_base_dir: str,
    area_threshold: float = DEFAULT_PERSON_AREA_THRESHOLD,
    return_vis: bool = True
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_base_dir, exist_ok=True)
    model = load_yolo_model(model_path)

    for label in ['pass', 'non-pass']:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)
        os.makedirs(os.path.join(vis_base_dir, label), exist_ok=True)

    json_files = [
        f for f in os.listdir(json_dir)
        if f.lower().endswith('.json')
    ]

    for fname in json_files:
        json_path = os.path.join(json_dir, fname)
        with open(json_path, 'r', encoding='utf-8') as jf:
            rec = json.load(jf)
        img_name = rec.get('filename')
        img_path = rec.get('filepath')

        if rec.get('pass') is False:
            continue

        if not img_path or not os.path.isfile(img_path):
            print(f'이미지 없음: {json_path}')
            continue

        result, vis_img, orig_img = process_single_image(model, img_path, area_threshold, return_vis)

        rec.update(result)
        save_result(rec, json_path)
        print(f'[저장됨] {json_path}')

        judgement_dir = 'pass' if result['pass'] else 'non-pass'

        copy_image(img_path, output_dir, judgement_dir)
        print(f'[이미지 복사됨] {img_name} → {output_dir}')

        vis_dir = os.path.join(vis_base_dir, judgement_dir)
        vis_save_path = os.path.join(vis_dir, img_name)

        if return_vis and vis_img is not None:
            cv2.imwrite(vis_save_path, vis_img.astype(np.uint8))
            print(f'[시각화 이미지 저장됨] {vis_dir}')
        else:
            cv2.imwrite(vis_save_path, orig_img)
            print(f'[원본 시각화 저장됨] {vis_save_path}')

        print(f'[결과] 사람 중심 여부: {result['person_dominant']}, 비율: {result['person_area_ratio']:.2%}\n')
