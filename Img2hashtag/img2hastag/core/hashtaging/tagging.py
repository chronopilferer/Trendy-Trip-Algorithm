import os
import json
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from openai import OpenAI
from img2hastag.utils.config import load_config

os.environ.pop("SSL_CERT_FILE", None)
_client = None

def init_client(api_key: str):
    global _client
    _client = OpenAI(api_key=api_key)

def encode_image(image_path: str) -> str:
    with Image.open(image_path) as img:
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

def generate_hashtags(model: str, detail: str, temperature: float, max_tokens: int,
                      prompt_text: str, image_path: str) -> str:
    if _client is None:
        raise RuntimeError("OpenAI client가 초기화되지 않았습니다. init_client() 호출 필요.")
    b64 = encode_image(image_path)
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "너는 이미지에 어울리는 감성적 해시태그를 만들어주는 어시스턴트야."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}}
                ]
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def update_json_with_hashtags(json_path: str, data: dict, hashtags: str):
    data["hashtags"] = hashtags
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def find_all_jsons(root_dir: str) -> list:
    return [str(p) for p in Path(root_dir).rglob("*.json")]

def extract_category(json_path: str, root_dir: str) -> str:
    rel = Path(json_path).relative_to(root_dir)
    return rel.parts[0] if rel.parts else ""

def process_json_file(json_path: str, config_hashtag: dict, root_dir: str):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    if data.get("pass_type") != "pass" or "hashtags" in data:
        return  

    img_path = data.get("file_path")
    if not img_path or not Path(img_path).is_file():
        return
    
    category = extract_category(json_path, root_dir)
    prompt_key = f"{category}_prompt"
    prompt_text = config_hashtag.get(prompt_key)
    if not prompt_text:
        return
    
    print(f"Processing {json_path} with prompt: {prompt_text}")

    hashtags = generate_hashtags(
        model=config_hashtag["model"],
        detail=config_hashtag["detail"],
        temperature=config_hashtag["temperature"],
        max_tokens=config_hashtag["max_tokens"],
        prompt_text=prompt_text,
        image_path=img_path
    )
    update_json_with_hashtags(json_path, data, hashtags)

def process_tagging(json_root_dir: str, config_path: str, openai_api_key: str):
    cfg = load_config(config_path).get("hashtag", {})
    init_client(openai_api_key)

    for jp in find_all_jsons(json_root_dir):
        try:
            process_json_file(jp, cfg, json_root_dir)
        except Exception:
            continue
