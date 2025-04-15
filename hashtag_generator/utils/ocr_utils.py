import easyocr
import cv2
import numpy as np
from typing import List

ocr = easyocr.Reader(['ko', 'en'], gpu=True)

def get_text_boxes(image: np.ndarray) -> List:
    result = ocr.readtext(image)
    boxes = []
    for item in result:
        box = item[0]
        if len(box) == 4:
            boxes.append(box)
        else:
            x_min = min(p[0] for p in box)
            y_min = min(p[1] for p in box)
            x_max = max(p[0] for p in box)
            y_max = max(p[1] for p in box)
            boxes.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    return boxes

def compute_text_area_ratio(image: np.ndarray, boxes: List) -> float:
    img_area = image.shape[0] * image.shape[1]
    total_box_area = 0
    for box in boxes:
        box_np = np.array([[int(p[0]), int(p[1])] for p in box], dtype=np.int32)
        area = cv2.contourArea(box_np)
        total_box_area += area
    return total_box_area / img_area if img_area > 0 else 0

def draw_text_boxes(image: np.ndarray, boxes: List) -> np.ndarray:
    vis_img = image.copy().astype(np.uint8)
    for box in boxes:
        pts = np.array([[int(p[0]), int(p[1])] for p in box], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
    return vis_img
