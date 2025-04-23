import json
import os
import shutil

def save_result(result: dict, output_path: str):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[파일 저장 실패] {output_path}: {e}")


def copy_image(img_path: str, filtered_base_dir: str, judgement: str):

    if not img_path or not os.path.exists(img_path):
        print(f"[이미지 없음] {img_path}")
        return
    
    filename = os.path.basename(img_path)
    target_dir = os.path.join(filtered_base_dir, judgement)
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(img_path, os.path.join(target_dir, filename))
    print(f"[이미지 복사됨] {img_path} → {target_dir}")
