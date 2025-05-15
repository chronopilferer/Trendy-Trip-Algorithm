# run_caption_llava.py
import logging
import yaml
import torch
from pathlib import Path

from img2hastag.core.filtering.llava_captioning import (
    process_captioning as llava_captioning,
    load_llava_model
)
from img2hastag.utils.logging import setup_logging

def main(config_path: str = "configs/config.yml"):
    setup_logging()
    logger = logging.getLogger(__name__)
    cfg = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        llava_proc, llava_model = load_llava_model(
            cfg['captioning']['llava_model'], device
        )
    except Exception as e:
        logger.error(f"[LLaVA 모델 로딩 실패] {e}", exc_info=True)
        return

    for category in ["cafe_img_dir", "restaurant_img_dir", "attraction_img_dir"]:
        json_dir = Path(cfg['path']['json_dir']) / Path(cfg['path'][category]).stem
        logger.info(f"[LLaVA Caption] {json_dir.stem}")
        llava_captioning(
            json_dir=str(json_dir),
            processor=llava_proc,
            model=llava_model,
            prompt=cfg['captioning']['llava_prompt'],
            device=device,
            max_new_tokens=cfg['captioning'].get('max_new_tokens', 128)
        )
        torch.cuda.empty_cache()

    del llava_model, llava_proc
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()