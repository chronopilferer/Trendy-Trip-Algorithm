from utils.yolo_utils import detect_objects_yolo, deny_food_only_classes, pass_scene_classes
import cv2
from typing import List

def is_food_only(labels: List[str]) -> bool:
    has_pass_obj = any(obj in pass_scene_classes for obj in labels)
    has_food_obj = any(obj in deny_food_only_classes for obj in labels)
    return has_food_obj and not has_pass_obj

def is_food_only_image(image_path: str) -> bool:
    image = cv2.imread(image_path)
    if image is None:
        return False
    labels = detect_objects_yolo(image)
    return is_food_only(labels)
