from img2hastag.core.hashtaging.tagging import process_tagging
import os

if __name__ == "__main__":
    json_root_dir = "./data/"            
    config_path = "./configs"            
    openai_api_key = os.getenv("OPENAI_API_KEY")  

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

    process_tagging(json_root_dir, config_path, openai_api_key)
