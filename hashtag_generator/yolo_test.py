from filtering.yolo_filter import is_food_only_image

import os
import shutil
from pathlib import Path

# 기본 경로
DATA_DIR = Path("./data")
IMAGE_DIR = DATA_DIR / "images_filtered" / "step-1" / "적합"
PASS_YOLO_DIR = DATA_DIR / "images_filtered" / "step-2" / "적합"
FAIL_YOLO_DIR = DATA_DIR / "images_filtered" / "step-2" / "부적합"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# 폴더 생성
PASS_YOLO_DIR.mkdir(parents=True, exist_ok=True)
FAIL_YOLO_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    for fname in os.listdir(IMAGE_DIR):
        fpath = IMAGE_DIR / fname
        ext = fpath.suffix.lower()

        if not fpath.is_file() or ext not in VALID_EXTENSIONS:
            continue

        result = is_food_only_image(str(fpath))
        print(f"[Result] 음식만 있는 이미지 여부: {result}")

        if result:
            shutil.copy(str(fpath), FAIL_YOLO_DIR / fname)
            print(f"[✗] 음식 전용 → 부적합/yolo/{fname}")
        else:
            shutil.copy(str(fpath), PASS_YOLO_DIR / fname)
            print(f"[✓] 통과 → 적합/yolo/{fname}")
