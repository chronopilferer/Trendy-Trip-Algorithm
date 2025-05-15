# run_preprocessing.py
import logging
import yaml
import torch
from pathlib import Path

from img2hastag.core.filtering.image import process_img_filtering
from img2hastag.core.filtering.yolo import process_yolo_filtering
from img2hastag.core.filtering.ocr import process_ocr_filtering
from img2hastag.core.filtering.clip import process_clip_filtering
from img2hastag.core.filtering.compute_stat import process_stat_compute
from img2hastag.core.filtering.stat_filtering import process_stat_filtering
from img2hastag.utils.logging import setup_logging

def main(config_path: str = "configs/config.yml"):
    setup_logging()
    logger = logging.getLogger(__name__)

    cfg = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    categories = ["cafe_img_dir", "restaurant_img_dir", "attraction_img_dir"]
    for category in categories:
        img_dir   = Path(cfg['path'][category])
        json_dir  = Path(cfg['path']['json_dir']) / img_dir.stem
        stats_dir = Path(cfg['path']['statistics_dir']) / img_dir.stem
        mid_dir   = Path(cfg['path']['mid_dir']) / img_dir.stem

        for d in (json_dir, stats_dir, mid_dir):
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Preprocess] {img_dir.stem}")
        process_img_filtering(json_dir=json_dir, data_dir=img_dir, category=img_dir.stem)
        process_yolo_filtering(json_dir=json_dir, data_dir=img_dir,
                              model_path=Path(cfg['yolo']['model_path']))
        process_ocr_filtering(json_dir=json_dir, data_dir=img_dir)
        process_clip_filtering(json_dir=json_dir, data_dir=img_dir,
                               model_name=cfg['clip']['model'], prompts=cfg['clip']['prompts'])
        process_stat_compute(json_dir=json_dir, output_dir=stats_dir,
                             fields=cfg.get('statistics', {}).get('fields'),
                             percentiles=cfg.get('statistics', {}).get('percentiles'))
        process_stat_filtering(json_dir=str(json_dir),
                               stats_csv=str(stats_dir / 'field_statistics.csv'),
                               output_dir=str(mid_dir))

    logger.info("[Preprocess] All categories done.")

if __name__ == '__main__':
    main()