import logging
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict

from utils.constants import TARGET_LABELS, VALID_EXTENSIONS
from utils.file_io import save_result
from utils.io import load_record

logger = logging.getLogger(__name__)

def load_yolo_model(model_path: Path) -> YOLO:
    """
    YOLO 모델을 로드하여 반환합니다.
    """
    return YOLO(str(model_path))

def detect_objects(model: YOLO, image: np.ndarray) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    """
    이미지에서 객체를 검출하여 (라벨, 바운딩 박스) 리스트를 반환합니다.
    """
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
    """
    이미지 전체 대비 사람(person)과 TARGET_LABELS 객체 면적 비율을 계산합니다.
    """
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
    """
    1) YOLO 모델 로드
    2) data_dir 내 이미지 파일 순회
    3) detect_objects -> compute_area_ratios
    4) JSON에 결과 누적 저장
    실패 시 로깅 후 건너뜁니다.
    """
    json_dir.mkdir(parents=True, exist_ok=True)
    model = load_yolo_model(model_path)

    for img_path in data_dir.iterdir():
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        try:
            # JSON 로드 또는 기본값 생성
            json_path = json_dir / f"{img_path.stem}.json"
            defaults = {'file_path': str(img_path)}
            rec = load_record(json_path, defaults)
            if "file_name" not in rec:
                rec["file_name"] = img_path.stem

            # 이미지 로드
            img = cv2.imread(str(img_path))
            if img is None:
                raise IOError(f"이미지 로드 실패: {img_path}")

            # 객체 검출 및 면적 계산
            labels, boxes = detect_objects(model, img)
            ratios = compute_area_ratios(labels, boxes, img.shape[:2])

            # 기록 업데이트 및 저장
            rec.update(ratios)
            save_result(rec, str(json_path))
            logger.info(f"Saved YOLO metrics for {img_path.name}")

        except Exception as e:
            logger.error(f"Failed YOLO processing {img_path.name}: {e}", exc_info=True)
