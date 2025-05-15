import logging
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict

from img2hastag.utils.constants import TARGET_LABELS, VALID_EXTENSIONS
from img2hastag.utils.io import save_result, load_record

logger = logging.getLogger(__name__)

def load_yolo_model(model_path: Path) -> YOLO:
    return YOLO(str(model_path))

def detect_objects(model: YOLO, image: np.ndarray) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    results = model(image, verbose=False)[0]
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    labels = [model.names[cid] for cid in cls_ids]
    bboxes = [tuple(box) for box in results.boxes.xyxy.cpu().numpy().astype(int)]
    return labels, bboxes

def compute_area_ratios(
    labels: List[str],
    boxes: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int]
) -> Dict[str, float]:
    h, w = image_shape
    total_area = max(h * w, 1)
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

def process_yolo_filtering(
    json_dir: Path,
    data_dir: Path,
    model_path: Path
) -> None:
    json_dir.mkdir(parents=True, exist_ok=True)
    model = load_yolo_model(model_path)

    for img_path in data_dir.iterdir():
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        try:
            json_path = json_dir / f"{img_path.stem}.json"
            rec = load_record(json_path, defaults={})

            img = cv2.imread(str(img_path))
            if img is None:
                raise IOError(f"이미지 로드 실패: {img_path}")

            labels, boxes = detect_objects(model, img)
            ratios = compute_area_ratios(labels, boxes, img.shape[:2])

            rec.update(ratios)
            save_result(rec, str(json_path))
            logger.info(f"Saved YOLO metrics for {img_path.name}")

        except Exception as e:
            logger.error(f"Failed YOLO processing {img_path.name}: {e}", exc_info=True)
