import os
import shutil
from filtering.rule_base_filter import image_filter_analysis
from pathlib import Path

# 기본 경로
DATA_DIR = Path("./data")
IMAGE_DIR = Path("./data/images_raw")
PASS_DIR = DATA_DIR / "images_filtered" / "step-1" / "적합"
FAIL_DIR = DATA_DIR / "images_filtered" / "step-1" / "부적합"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

PASS_DIR.mkdir(parents=True, exist_ok=True)

# 이미지 순회
for fname in os.listdir(IMAGE_DIR):
    fpath = IMAGE_DIR / fname
    ext = fpath.suffix.lower()

    if not fpath.is_file() or ext not in VALID_EXTENSIONS:
        continue

    result = image_filter_analysis(str(fpath))

    if result["status"] == "pass":
        # 적합 폴더로 복사
        shutil.copy(str(fpath), str(PASS_DIR / fname))
        print(f"[✓] 적합 → {fname}")
    else:
        # 실패 사유에 따라 폴더 생성
        reason = result.get("reason", "기타")
        fail_dir = FAIL_DIR / reason
        fail_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(fpath), str(fail_dir / fname))
        print(f"[✗] 부적합({reason}) → {fname}")
