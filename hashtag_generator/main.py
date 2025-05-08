import logging
from pathlib import Path

from utils.config import load_config

from modules.image import process_img_filtering
from modules.yolo import process_yolo_filtering
from modules.ocr import process_ocr_filtering
from modules.clip import process_clip_filtering
from modules.compute_stat import process_stat_compute
from modules.stat_filtering import process_stat_filtering
from modules.captioning import process_captioning
from modules.llm import process_llm_filtering

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def main():
    # 1) 설정 불러오기
    config = load_config("config")

    categories = ['cafe_img_dir', 'restaurant_img_dir', 'attraction_img_dir']

    for category in categories:

        img_dir         = Path(config["path"][category])
        json_dir        = Path(config["path"]["json_dir"]) / img_dir.stem
        output_dir      = Path(config["path"]["output_dir"]) / img_dir.stem
        stats_dir       = Path(config["path"]["statistics_dir"]) / img_dir.stem
        filter_dir      = output_dir / "mid"

        # 2) 로깅 설정
        setup_logging()
        logger = logging.getLogger(__name__)

        # 3) 필수 디렉토리 생성
        ensure_dirs(img_dir, json_dir, output_dir, stats_dir, filter_dir)
        logger.info("필수 디렉토리 준비 완료")

        # 4) 파이프라인 단계 실행
        try:
            # # 4.1) 이미지 메트릭 계산
            # logger.info("[1/7] 이미지 메트릭 계산 시작")
            # process_img_filtering(json_dir=json_dir, data_dir=img_dir)

            # # 4.2) YOLO 객체 검출
            # logger.info("[2/7] YOLO 객체 검출 시작")
            # process_yolo_filtering(
            #     json_dir=json_dir,
            #     data_dir=img_dir,
            #     model_path=config["yolo"]["model_path"],
            # )

            # # 4.3) OCR 메트릭 계산
            # logger.info("[3/7] OCR 메트릭 계산 시작")
            # process_ocr_filtering(json_dir=json_dir, data_dir=img_dir)

            # # 4.4) CLIP 점수 계산
            # logger.info("[4/7] CLIP 점수 계산 시작")
            # process_clip_filtering(
            #     json_dir=json_dir,
            #     data_dir=img_dir,
            #     model_name=config["clip"]["model"],
            #     prompts=config["clip"]["prompts"],
            # )

            # # 4.5) 통계량 계산
            # logger.info("[5/7] 통계량 계산 시작")
            # stat_fields      = config.get("statistics", {}).get("fields")
            # stat_percentiles = config.get("statistics", {}).get("percentiles")
            # process_stat_compute(
            #     json_dir=json_dir,
            #     output_dir=stats_dir,
            #     fields=stat_fields,
            #     percentiles=stat_percentiles,
            # )

            # 4.6) 통계 기반 필터링
            logger.info("[6/7] 통계 기반 필터링 시작")
            process_stat_filtering(
                json_dir=str(json_dir),
                stats_csv=str(stats_dir / "field_statistics.csv"),
                output_dir=str(filter_dir),
            )

            # # 4.7) 캡션 생성
            # logger.info("[7/7] 캡션 생성 및 LLM 필터링")
            # process_captioning(
            #     json_dir=json_dir,
            #     data_dir=img_dir,
            #     output_dir=output_dir,
            #     model_name=config["captioning"]["model"],
            #     prompts=config["captioning"]["prompt"],
            # )

            # # 4.8) LLM 필터링
            # process_llm_filtering(
            #     json_dir=json_dir,
            #     data_dir=img_dir,
            #     output_dir=output_dir,
            #     model_name=config["LLM"]["model"],
            #     prompts=config["LLM"]["prompt"],
            # )

            logger.info("파이프라인 실행 완료")

        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()