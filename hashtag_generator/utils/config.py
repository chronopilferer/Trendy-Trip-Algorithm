import os
import yaml

def load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    config_path = os.path.join(base_dir, "config", "config.yml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config
