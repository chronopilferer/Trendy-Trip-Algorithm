import yaml
from pathlib import Path

def load_config(config_dir: str = None, config_filename: str = "config.yml") -> dict:
    if config_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_dir = base_dir / "config"
    else:
        config_dir = Path(config_dir)  

    config_path = config_dir / config_filename

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
