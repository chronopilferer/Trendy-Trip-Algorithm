import logging
import yaml
import torch
from pathlib import Path

from Img2hashtag.img2hastag.core.filtering.captioning import process_captioning

from Img2hashtag.img2hastag.utils.logging import setup_logging

def main(config_path: str = "configs/config.yml"):
    setup_logging()
    logger = logging.getLogger(__name__)
    cfg = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blip_proc, blip_model = process_captioning.__globals__['load_img_to_text_model'](
        cfg['captioning']['instructblip_model'], device)

    for category in ["cafe_img_dir", "restaurant_img_dir", "attraction_img_dir"]:
        json_dir = Path(cfg['path']['json_dir']) / Path(cfg['path'][category]).stem
        logger.info(f"[BLIP Caption] {json_dir.stem}")
        process_captioning(json_dir=str(json_dir), processor=blip_proc, model=blip_model,
                           prompt=cfg['captioning']['instructblip_prompt'], device=device,
                           max_new_tokens=cfg['captioning'].get('max_new_tokens', 128))

        del blip_model, blip_proc
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()