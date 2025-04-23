from PIL import Image
import numpy as np
import cv2

def read_image_unicode_safe(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)