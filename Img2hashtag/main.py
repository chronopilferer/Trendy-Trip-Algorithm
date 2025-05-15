import logging
from pathlib import Path
import torch

from img2hastag.utils.config import load_config
from img2hastag.utils.io import ensure_dirs
from img2hastag.utils.logging import setup_logging

from img2hastag.core.filtering.captioning import load_img_to_text_model, process_captioning
from img2hastag.core.filtering.yolo import process_yolo_filtering
from img2hastag.core.filtering.ocr import process_ocr_filtering
from img2hastag.core.filtering.clip import process_clip_filtering
from img2hastag.core.filtering.compute_stat import process_stat_compute
from img2hastag.core.filtering.stat_filtering import process_stat_filtering
from img2hastag.core.filtering.image import process_img_filtering
from img2hastag.core.filtering.llm import process_llm_filtering

def main():
    # 1) 설정 불러오기
    config = load_config("img2hastag/config")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_img_to_text_model(
        config["captioning"]["instructblip_model"],
        device=device
    )

    image_source_folders = ['cafe_img_dir', 'restaurant_img_dir', 'attraction_img_dir']

    for image_source_category in image_source_folders:

        img_dir         = Path(config["path"][image_source_category])
        json_dir        = Path(config["path"]["json_dir"]) / img_dir.stem
        output_dir      = Path(config["path"]["output_dir"]) / img_dir.stem
        stats_dir       = Path(config["path"]["statistics_dir"]) / img_dir.stem
        filter_dir = output_dir / "mid" / img_dir.stem

        # 2) 로깅 설정
        setup_logging()
        logger = logging.getLogger(__name__)

        # 3) 필수 디렉토리 생성
        ensure_dirs(img_dir, json_dir, output_dir, stats_dir, filter_dir)
        logger.info("필수 디렉토리 준비 완료")

        category = img_dir.stem

        # 4) 파이프라인 단계 실행
        try:
            # 4.1) 이미지 메트릭 계산
            logger.info("[1/8] 이미지 메트릭 계산 시작")
            process_img_filtering(json_dir=json_dir, data_dir=img_dir, category=category)

            # 4.2) YOLO 객체 검출
            logger.info("[2/8] YOLO 객체 검출 시작")
            process_yolo_filtering(
                json_dir=json_dir,
                data_dir=img_dir,
                model_path=config["yolo"]["model_path"],
            )

            # 4.3) OCR 메트릭 계산
            logger.info("[3/8] OCR 메트릭 계산 시작")
            process_ocr_filtering(json_dir=json_dir, data_dir=img_dir)

            # 4.4) CLIP 점수 계산
            logger.info("[4/8] CLIP 점수 계산 시작")
            process_clip_filtering(
                json_dir=json_dir,
                data_dir=img_dir,
                model_name=config["clip"]["model"],
                prompts=config["clip"]["prompts"],
            )

            # 4.5) 통계량 계산
            logger.info("[5/8] 통계량 계산 시작")
            stat_fields      = config.get("statistics", {}).get("fields")
            stat_percentiles = config.get("statistics", {}).get("percentiles")
            process_stat_compute(
                json_dir=json_dir,
                output_dir=stats_dir,
                fields=stat_fields,
                percentiles=stat_percentiles,
            )

            # 4.6) 통계 기반 필터링
            logger.info("[6/8] 통계 기반 필터링 시작")
            process_stat_filtering(
                json_dir=str(json_dir),
                stats_csv=str(stats_dir / "field_statistics.csv"),
                output_dir=str(filter_dir),
            )

            # 4.7) 캡션 생성
            logger.info("[7/8] 캡션 생성 시작")
            process_captioning(
                json_dir=str(json_dir),
                processor=processor,
                model=model,
                prompt=config["captioning"]["instructblip_prompt"],
                device=device,
                max_new_tokens=128
            )

            # 4.8) LLM 필터링
            logger.info("[8/8] LLM 필터링 시작")
            process_llm_filtering(
                json_dir=str(filter_dir),
                output_dir=str(output_dir),
                model_id=config["LLM"]["model"],
                prompt_template=config["LLM"]["prompt_template"],
                max_new_tokens=config["LLM"].get("max_new_tokens", 10),
                temperature=config["LLM"].get("temperature", 0.2),
                top_p=config["LLM"].get("top_p", 0.95),
                load_in_4bit=config["LLM"].get("load_in_4bit", True),
                device_map=config["LLM"].get("device_map", "auto"),
                torch_dtype=torch.bfloat16,
            )

            logger.info("파이프라인 실행 완료")

        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()