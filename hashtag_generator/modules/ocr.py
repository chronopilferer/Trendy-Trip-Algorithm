import logging
from pathlib import Path
import cv2
import numpy as np
import easyocr
from typing import List, Tuple, Dict

from utils.constants import VALID_EXTENSIONS
from utils.file_io import save_result
from utils.io import load_record

logger = logging.getLogger(__name__)

def load_ocr_model(langs: List[str] = ['ko', 'en'], gpu: bool = True) -> easyocr.Reader:
    """
    EasyOCR 리더를 초기화하여 반환합니다.
    """
    return easyocr.Reader(lang_list=langs, gpu=gpu)

def detect_text_boxes(
    reader: easyocr.Reader,
    image: np.ndarray
) -> List[List[Tuple[int, int]]]:
    """
    이미지에서 검출된 텍스트 박스 좌표 리스트를 반환합니다.
    """
    results = reader.readtext(image)
    boxes: List[List[Tuple[int, int]]] = []
    for coords, _, _ in results:
        if len(coords) == 4:
            pts = [(int(x), int(y)) for x, y in coords]
        else:
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            pts = [
                (int(min(xs)), int(min(ys))),
                (int(max(xs)), int(min(ys))),
                (int(max(xs)), int(max(ys))),
                (int(min(xs)), int(max(ys)))
            ]
        boxes.append(pts)
    return boxes

def compute_text_area_ratio(
    boxes: List[List[Tuple[int, int]]],
    image_shape: Tuple[int, int]
) -> float:
    """
    텍스트 박스 면적 합계를 이미지 전체 면적으로 나눈 비율을 반환합니다.
    """
    h, w = image_shape
    total_area = max(h * w, 1)
    text_area = 0.0
    for box in boxes:
        contour = np.array(box, dtype=np.int32)
        text_area += cv2.contourArea(contour)
    return float(text_area / total_area)


def process_single_ocr(fpath: Path, reader: easyocr.Reader) -> Dict[str, float]:
    """
    단일 이미지 파일에 대해 OCR을 수행하고, 텍스트 면적 비율과 박스 개수를 반환합니다.
    """
    img = cv2.imread(str(fpath))
    if img is None:
        raise IOError(f"이미지 로드 실패: {fpath}")

    boxes = detect_text_boxes(reader, img)
    ratio = compute_text_area_ratio(boxes, img.shape[:2])
    return {
        'text_area_ratio': ratio,
        'num_text_boxes': float(len(boxes))
    }

def process_ocr_filtering(json_dir: Path, data_dir: Path) -> None:
    """
    이미지 디렉토리를 순회하며 OCR 메트릭을 JSON 파일에 누적 저장합니다.
    실패 파일은 로깅 후 건너뜁니다.
    """
    json_dir.mkdir(parents=True, exist_ok=True)
    reader = load_ocr_model()

    for img_path in data_dir.iterdir():
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        try:
            json_path = json_dir / f"{img_path.stem}.json"
            defaults = {'file_path': str(img_path)}
            rec = load_record(json_path, defaults)
            if "file_name" not in rec:
                rec["file_name"] = img_path.stem

            metrics = process_single_ocr(img_path, reader)
            rec.update(metrics)

            save_result(rec, str(json_path))
            logger.info(f"Saved OCR metrics for {img_path.name}")
        except Exception as e:
            logger.error(f"Failed OCR processing {img_path.name}: {e}", exc_info=True)
