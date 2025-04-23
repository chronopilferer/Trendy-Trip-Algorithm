from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Tuple
import os
from hashtag_generator.utils.config import load_config

# 모델 로딩
config = load_config()
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config["model"]["yolo_path"])
model = YOLO(model_path)

# 사람 비율 기준
PERSON_AREA_THRESHOLD = 0.3

def detect_objects_yolo(image: np.ndarray) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    results = model(image, verbose=False)[0]
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    labels = [model.names[cid] for cid in class_ids]
    bboxes = results.boxes.xyxy.cpu().numpy().astype(int)
    return labels, bboxes.tolist()

def is_person_dominant(labels: List[str], boxes: List[Tuple[int, int, int, int]], image_shape: Tuple[int, int]) -> bool:
    h, w = image_shape[:2]
    total_area = h * w
    person_area = sum((x2 - x1) * (y2 - y1) for label, (x1, y1, x2, y2) in zip(labels, boxes) if label == "person")
    return person_area / total_area > PERSON_AREA_THRESHOLD if total_area > 0 else False

def draw_bounding_boxes(image: np.ndarray, labels: List[str], boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    vis_img = image.copy().astype(np.uint8)
    for label, (x1, y1, x2, y2) in zip(labels, boxes):
        if label == "person":
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis_img

def is_person_dominant_image(image_path: str, area_threshold: float = 0.3, return_vis: bool = True) -> Tuple[bool, float, np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] 이미지 로딩 실패: {image_path}")
        return False, 0.0, None

    labels, boxes = detect_objects_yolo(image)
    h, w = image.shape[:2]
    total_area = h * w
    person_area = sum((x2 - x1) * (y2 - y1) for label, (x1, y1, x2, y2) in zip(labels, boxes) if label == "person")
    ratio = person_area / total_area if total_area > 0 else 0

    flag = ratio > area_threshold
    vis_img = draw_bounding_boxes(image, labels, boxes) if return_vis else None
    return flag, ratio, vis_img
