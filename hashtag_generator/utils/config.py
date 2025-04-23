import os
import yaml

def load_config(config_dir: str = None, config_filename: str = "config.yml") -> dict:
    if config_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(base_dir, "config")

    config_path = os.path.join(config_dir, config_filename)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
