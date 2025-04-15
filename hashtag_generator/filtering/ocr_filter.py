from utils.ocr_utils import get_text_boxes, compute_text_area_ratio, draw_text_boxes
from PIL import Image
import numpy as np

def is_text_dominant(image_path: str, area_threshold: float = 0.2):
    try:
        with Image.open(image_path) as img:
            image = np.array(img.convert("RGB"))
    except Exception as e:
        print(f"[ERROR] 이미지 로딩 실패: {image_path}, 에러: {e}")
        return False, 0.0, None

    boxes = get_text_boxes(image)
    print(f"Detected {len(boxes)} box(es) in {image_path}")
    if not boxes:
        print(f"[!] 텍스트 박스 감지 실패: {image_path}")

    vis_img = draw_text_boxes(image.copy(), boxes)
    ratio = compute_text_area_ratio(image, boxes)
    return ratio > area_threshold, ratio, vis_img
