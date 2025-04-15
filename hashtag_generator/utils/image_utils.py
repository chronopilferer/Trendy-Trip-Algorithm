import cv2
import numpy as np
from PIL import Image
from typing import Tuple

def read_image_unicode_safe(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def is_too_dark_or_bright(image: np.ndarray, dark_thresh: int = 40, bright_thresh: int = 220) -> str:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    if brightness < dark_thresh:
        return "too_dark"
    elif brightness > bright_thresh:
        return "too_bright"
    return "ok"

def is_low_resolution(image: np.ndarray, min_width: int = 300, min_height: int = 300) -> bool:
    h, w = image.shape[:2]
    return w < min_width or h < min_height

def is_low_entropy(image: np.ndarray, entropy_thresh: float = 3.5) -> Tuple[bool, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    entropy = -np.sum([p * np.log2(p) for p in hist_norm if p > 0])
    return entropy < entropy_thresh, entropy
