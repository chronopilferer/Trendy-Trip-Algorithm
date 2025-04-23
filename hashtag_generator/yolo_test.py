from utils.yolo_utils import is_person_dominant_image
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import cv2  # 꼭 필요함!

# 경로 설정
DATA_DIR = Path("./data")
IMAGE_DIR = DATA_DIR / "images_filtered" / "step-1" / "적합"
PASS_YOLO_DIR = DATA_DIR / "images_filtered" / "step-2" / "적합"
FAIL_YOLO_DIR = DATA_DIR / "images_filtered" / "step-2" / "부적합"
PASS_VIS_DIR = PASS_YOLO_DIR / "yolo_vis"
FAIL_VIS_DIR = FAIL_YOLO_DIR / "yolo_vis"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# 폴더 생성
for dir_path in [PASS_YOLO_DIR, FAIL_YOLO_DIR, PASS_VIS_DIR, FAIL_VIS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for fname in os.listdir(IMAGE_DIR):
        fpath = IMAGE_DIR / fname
        ext = fpath.suffix.lower()

        if not fpath.is_file() or ext not in VALID_EXTENSIONS:
            continue

        flag, ratio, vis_img = is_person_dominant_image(str(fpath), area_threshold=0.15, return_vis=True)

        if flag:
            target_img_dir = FAIL_YOLO_DIR
            target_vis_dir = FAIL_VIS_DIR
            print(f"[✗] 사람 과다 → 부적합/yolo/{fname}")
        else:
            target_img_dir = PASS_YOLO_DIR
            target_vis_dir = PASS_VIS_DIR
            print(f"[✓] 통과 → 적합/yolo/{fname}")

        shutil.copy(str(fpath), target_img_dir / fname)

        if vis_img is not None:
            save_path = target_vis_dir / fname
            try:
                # 🔥 핵심! BGR → RGB로 변환 후 저장
                vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                vis_pil = Image.fromarray(vis_rgb)
                vis_pil.save(str(save_path))
                print(f"[💾 저장 결과] {save_path} → 성공")
            except Exception as e:
                print(f"[❌ PIL 저장 실패] {save_path} → {e}")
        else:
            print(f"[⚠️ 시각화 없음] {fname}")

        print(f"[결과] 사람 중심 여부: {flag}, 비율: {ratio:.2%}\n")
