import os
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict, Union
import easyocr

from utils.draw import draw_boxes
from utils.constants import DEFAULT_TEXT_AREA_THRESHOLD
from utils.file_io import save_result, copy_image

def load_ocr_model(langs: List[str] = ['ko', 'en'], gpu: bool = True) -> easyocr.Reader:
    return easyocr.Reader(langs, gpu=gpu)

def detect_text(model: easyocr.Reader, image: np.ndarray) -> List[List[Tuple[float, float]]]:
    results = model.readtext(image)
    boxes: List[List[Tuple[float, float]]] = []

    for coords, _, _ in results:
        if len(coords) == 4:
            boxes.append(coords)
        else:
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            boxes.append([
                (min(xs), min(ys)), (max(xs), min(ys)),
                (max(xs), max(ys)), (min(xs), max(ys))
            ])
    return boxes

def is_text_dominant(
    boxes: List[List[Tuple[float, float]]],
    image_shape: Tuple[int, int],
    threshold: float = DEFAULT_TEXT_AREA_THRESHOLD
) -> Tuple[bool, float]:
    h, w = image_shape
    total_area = h * w
    if total_area == 0:
        return False, 0.0

    text_area = sum(
        cv2.contourArea(np.array([[int(x), int(y)] for x, y in box], np.int32))
        for box in boxes
    )
    ratio = text_area / total_area
    return ratio > threshold, ratio

def process_single_image(
    model: easyocr.Reader,
    img_path: str,
    area_threshold: float,
    return_vis: bool
) -> Tuple[Dict, Union[np.ndarray, None], np.ndarray]:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'이미지 로딩 실패: {img_path}')

    boxes = detect_text(model, img)
    flag, ratio = is_text_dominant(boxes, img.shape[:2], area_threshold)
    vis = draw_boxes(img.copy(), boxes) if return_vis else None

    result = {
        'text_dominant': bool(flag),
        'text_area_ratio': float(ratio),
        'pass': bool(not flag)  
    }

    return result, vis, img

def process_ocr_filtering(
    json_dir: str,
    output_dir: str,
    vis_base_dir: str,
    area_threshold: float = DEFAULT_TEXT_AREA_THRESHOLD,
    return_vis: bool = True,
    langs: List[str] = ['ko', 'en']
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_base_dir, exist_ok=True)

    for label in ['pass', 'non-pass']:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)
        os.makedirs(os.path.join(vis_base_dir, label), exist_ok=True)

    reader = load_ocr_model(langs=langs)

    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]

    for fname in json_files:
        json_path = os.path.join(json_dir, fname)

        with open(json_path, 'r', encoding='utf-8') as jf:
            rec = json.load(jf)

        img_path = rec.get('filepath')
        img_name = rec.get('filename')

        if rec.get('pass') is False:
            continue

        if not img_path or not os.path.isfile(img_path):
            print(f'[이미지 없음] {json_path}')
            continue

        result, vis_img, orig_img = process_single_image(reader, img_path, area_threshold, return_vis)

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
            print(f'[시각화 이미지 저장됨] {vis_save_path}')
        else:
            cv2.imwrite(vis_save_path, orig_img)
            print(f'[원본 시각화 저장됨] {vis_save_path}')

        print(f'[결과] 텍스트 중심 여부: {result['text_dominant']}, 비율: {result['text_area_ratio']:.2%}\n')