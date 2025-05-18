import logging
import yaml
import torch
from pathlib import Path

from img2hastag.core.filtering.llm_score import process_llm_filtering, load_filtering_model
from img2hastag.utils.logging import setup_logging

def main(config_path: str = "configs/config.yml"):
    setup_logging()
    logger = logging.getLogger(__name__)
    cfg = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = load_filtering_model(
        model_id=cfg['LLM']['model'],
        load_in_4bit=cfg['LLM'].get('load_in_4bit', True),
        device_map=cfg['LLM'].get('device_map', 'auto'),
        torch_dtype=getattr(torch, cfg['LLM'].get('torch_dtype', 'bfloat16'))
    )

    for category in ["cafe_img_dir", "restaurant_img_dir", "attraction_img_dir"]:
        category_name = Path(cfg['path'][category]).name  
        json_dir = Path(cfg['path']['final_dir']) / category_name / 'json'
        output_dir = json_dir.parent / 'score'
        logger.info(f"[LLM Filter] {json_dir}")
        
        process_llm_filtering(
            json_dir=str(json_dir),
            output_dir=str(output_dir),
            tokenizer=tokenizer,               
            model=model,
            device=device,
            prompt_template=cfg['LLM']['prompt_score'],
            max_new_tokens=cfg['LLM']['max_new_tokens'],
            temperature=cfg['LLM']['temperature'],
            top_p=cfg['LLM']['top_p'],
        )
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()