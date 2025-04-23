import os
import json
import cv2
import numpy as np
from typing import Dict

from utils.constants import (
    MIN_WIDTH,
    MIN_HEIGHT,
    DARKNESS_THRESHOLD,
    BRIGHTNESS_THRESHOLD,
    ENTROPY_THRESHOLD,
    VALID_EXTENSIONS
)
from utils.file_io import copy_image, save_result

def is_too_dark_or_bright(
    image: np.ndarray,
    dark_thresh: int = DARKNESS_THRESHOLD,
    bright_thresh: int = BRIGHTNESS_THRESHOLD
) -> str:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    if brightness < dark_thresh:
        return 'too_dark'
    elif brightness > bright_thresh:
        return 'too_bright'
    return 'ok'

def is_low_resolution(
    image: np.ndarray,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT
) -> bool:
    h, w = image.shape[:2]
    return w < min_width or h < min_height

def is_low_entropy(
    image: np.ndarray,
    entropy_thresh: float = ENTROPY_THRESHOLD
) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    entropy = -np.sum([p * np.log2(p) for p in hist_norm if p > 0])
    return entropy < entropy_thresh

def process_single_image_filtering(
    fname: str,
    fpath: str
) -> Dict:
    image = cv2.imread(fpath)
    if image is None:
        return None

    bright_status = is_too_dark_or_bright(image)
    is_low_res = is_low_resolution(image)
    is_low_ent = is_low_entropy(image)

    passed = (bright_status == 'ok') and not is_low_res and not is_low_ent

    result = {
        'filename': fname,
        'filepath': fpath,
        'brightness': bright_status,
        'is_low_resolution': bool(is_low_res),
        'is_low_entropy': bool(is_low_ent),
        'pass': bool(passed)
    }

    return result

def process_img_filtering(
    data_dir: str,
    output_dir: str,
    json_dir: str
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    for label in ['pass', 'non-pass']:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    results = []

    for img_file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_file)
        fname = img_file.split('.')[-2]

        if not os.path.isfile(img_path) or not img_file.lower().endswith(VALID_EXTENSIONS):
            continue

        result = process_single_image_filtering(img_file, img_path)
        if result is None:
            continue

        results.append(result)

        judgement_dir = 'pass' if result['pass'] else 'non-pass'

        copy_image(img_path, output_dir, judgement_dir)
        json_file = fname + '.json'
        json_path = os.path.join(json_dir, json_file)

        save_result(result, json_path)
        print(f'[저장됨] {json_path}')