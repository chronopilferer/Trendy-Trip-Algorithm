from filtering.ocr_filter import is_text_dominant
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# 기본 경로 설정
DATA_DIR = Path("./data")
IMAGE_DIR = DATA_DIR / "images_filtered" / "step-2" / "적합"
PASS_OCR_DIR = DATA_DIR / "images_filtered" / "step-3" / "적합"
FAIL_OCR_DIR = DATA_DIR / "images_filtered" / "step-3" / "부적합"

# 시각화 이미지 저장 경로 (원본과 별도로 저장)
PASS_VIS_DIR = PASS_OCR_DIR / "ocr_vis"
FAIL_VIS_DIR = FAIL_OCR_DIR / "ocr_vis"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# 저장할 폴더들 생성
PASS_OCR_DIR.mkdir(parents=True, exist_ok=True)
FAIL_OCR_DIR.mkdir(parents=True, exist_ok=True)
PASS_VIS_DIR.mkdir(parents=True, exist_ok=True)
FAIL_VIS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for fname in os.listdir(IMAGE_DIR):
        fpath = IMAGE_DIR / fname
        ext = fpath.suffix.lower()
        if not fpath.is_file() or ext not in VALID_EXTENSIONS:
            continue

        # OCR 필터 실행 → flag, 비율, 텍스트 박스가 그려진 이미지 반환
        flag, ratio, vis_img = is_text_dominant(str(fpath), area_threshold=0.1)

        # flag 값에 따라 적합과 부적합 경로를 결정
        if flag:
            # 텍스트가 과도하면 부적합으로 분류
            target_img_dir = FAIL_OCR_DIR
            target_vis_dir = FAIL_VIS_DIR
            print(f"[✗] 텍스트 과다 → 부적합/ocr/{fname}")
        else:
            target_img_dir = PASS_OCR_DIR
            target_vis_dir = PASS_VIS_DIR
            print(f"[✓] 통과 → 적합/ocr/{fname}")

        # 원본 이미지 (텍스트 박스 없는)를 복사하여 저장
        shutil.copy(str(fpath), target_img_dir / fname)

        # 시각화 이미지 (텍스트 박스 그려진 이미지)를 저장
        if vis_img is not None:
            os.makedirs(target_vis_dir, exist_ok=True)
            save_path = target_vis_dir / fname
            print(f"[🖼️ 이미지 타입] type: {type(vis_img)}, shape: {getattr(vis_img, 'shape', None)}")
            print(f"[📄 저장 경로] {save_path}")

            try:
                vis_pil = Image.fromarray(vis_img.astype(np.uint8))
                vis_pil.save(str(save_path))
                print(f"[💾 저장 결과] {save_path} → 성공 (PIL)")
            except Exception as e:
                print(f"[❌ PIL 저장 실패] {save_path} → {e}")
        else:
            with Image.open(fpath) as img:
                img_rgb = np.array(img.convert("RGB"))
                os.makedirs(target_vis_dir, exist_ok=True)
                Image.fromarray(img_rgb).save(str(target_vis_dir / fname))


        print(f"[결과] 텍스트 중심 여부: {flag}, 비율: {ratio:.2%}")
