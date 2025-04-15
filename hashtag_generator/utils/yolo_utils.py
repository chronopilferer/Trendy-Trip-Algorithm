from ultralytics import YOLO
import numpy as np
from typing import List
import os

from utils.config_loader import load_config

config = load_config()
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          config["model"]["yolo_path"])

model = YOLO(model_path)

deny_food_only_classes = {
    'pizza', 'cake', 'sandwich', 'donut', 'hot dog', 'bowl',
    'banana', 'apple', 'orange', 'broccoli', 'carrot', 'cup',
    'spoon', 'fork', 'knife'
}
pass_scene_classes = {
    'person', 'chair', 'dining table', 'couch', 'tv',
    'potted plant', 'bed', 'bench', 'sink'
}

def detect_objects_yolo(image: np.ndarray) -> List[str]:
    results = model(image, verbose=False)[0]
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    labels = [model.names[cid] for cid in class_ids]
    return labels
