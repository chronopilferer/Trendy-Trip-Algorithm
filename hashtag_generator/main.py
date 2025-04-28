import os

from utils.config import load_config
from utils.constants import DEFAULT_PERSON_AREA_THRESHOLD, DEFAULT_TEXT_AREA_THRESHOLD

from modules.captioning import process_captioning
from modules.ocr import process_ocr_filtering
from modules.yolo import process_yolo_filtering
from modules.image import process_img_filtering
from modules.llm import process_llm_filtering

def main():
    # 1) 설정 불러오기
    config = load_config("config")
    raw_img_dir = config['path']["img_dir"]       # 원본 이미지 저장 폴더
    json_dir = config['path']["json_dir"]      # 단계별 JSON 저장 폴더
    output_dir = config['path']["output_dir"]    # 결과 저장 폴더

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # 2) 1단계: 이미지 전처리 필터링 → pass 디렉토리
    # step1_output_dir = os.path.join(output_dir, "step_1")
    # step1_base = raw_img_dir
    # process_img_filtering(
    #     data_dir = step1_base,
    #     output_dir = step1_output_dir,
    #     json_dir = json_dir
    # )

    # 3) 2단계: YOLO 필터링 → pass 디렉토리
    # step2_output_dir = os.path.join(output_dir, "step_2")
    # process_yolo_filtering(
    #     json_dir = json_dir,
    #     output_dir = step2_output_dir,
    #     model_path = config["yolo"]["model_path"],
    #     vis_base_dir = os.path.join(step2_output_dir, "vis"),
    #     return_vis = config["yolo"]['return_vis']
    # )

    # 4) 3단계: OCR 필터링 → pass 디렉토리
    # step3_output_dir = os.path.join(output_dir, "step_3")
    # process_ocr_filtering(
    #     json_dir = json_dir,
    #     output_dir = step3_output_dir,
    #     vis_base_dir = os.path.join(step3_output_dir, "vis"),
    #     langs = config["ocr"]["langs"],
    #     return_vis = config["ocr"]['return_vis']
    # )

    # 5) 4단계: 캡션 생성 (이미지 이동 없음, JSON만)
    process_captioning(
        json_dir = json_dir,
        model_name = config['captioning']["model"],
        prompt = config['captioning']["prompt"]
    )

    # 6) 5단계: LLM 필터링 → pass 디렉토리
    step5_output_dir = os.path.join(output_dir, "step_4")
    process_llm_filtering(
        json_dir         = json_dir,
        output_dir       = step5_output_dir,
        model_id         = config['LLM']['model'],
        prompt_template  = config['LLM']['prompt_template'],
        max_new_tokens   = config['LLM']['max_new_tokens'],
        temperature      = config['LLM']['temperature'],
        top_p            = config['LLM']['top_p'],
    )

if __name__ == "__main__":
    main()