import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from utils.config import load_config
os.environ.pop("SSL_CERT_FILE", None)  

from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

def generate_hashtags(config, image_path):
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=config['hashtag']["model"],
        messages=[
            {"role": "system", "content": "너는 이미지에 어울리는 감성적 해시태그를 만들어주는 어시스턴트야."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config['hashtag']["cafe_prompt"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": config['hashtag']["detail"]
                        }
                    }
                ]
            }
        ],
        temperature=config['hashtag']["temperature"],
        max_tokens=config['hashtag']["max_tokens"]
    )

    return response.choices[0].message.content.strip()

def save_result(result, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"✅ 해시태그가 {path}에 저장되었습니다.")

if __name__ == "__main__":
    config = load_config()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ OPENAI_API_KEY가 누락되었습니다.")

    img_path = './data/images_raw/1.5커피_2.jpg'
    try:
        result = generate_hashtags(config, img_path)
        print("🔍 생성된 해시태그:\n", result)
        fname = os.path.basename(img_path)
        out_path = f"./data/hashtags/{config['hashtag']['model']}-{fname.split('.jpg')[0]}.txt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_result(result, out_path)
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
