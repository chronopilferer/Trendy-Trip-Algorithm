# run_pipeline.py
import gc
import logging
import torch

from img2hastag.utils.logging import setup_logging
from run_preprocessing import main as preprocess_all
from run_caption_blip import main as caption_blip
from run_caption_llava import main as caption_llava
from run_llm_filtering import main as llm_filter
from run_score import main as llm_score

def main(config_path: str = "configs/config.yml"):
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("==== 전체 파이프라인 시작 ====")

    # 1) 전처리
    # preprocess_all(config_path)

    # # 2) BLIP 캡션
    # caption_blip(config_path)
    # gc.collect()
    # torch.cuda.empty_cache()

    # # 3) LLaVA 캡션
    # caption_llava(config_path)
    # gc.collect()
    # torch.cuda.empty_cache()

    # 4) LLM 필터링
    llm_filter(config_path)
    gc.collect()
    torch.cuda.empty_cache()

    # # 5) LLM 점수 계산
    # llm_score(config_path)
    # gc.collect()
    # torch.cuda.empty_cache()

    logger.info("==== 전체 파이프라인 완료 ====")

if __name__ == "__main__":
    main()