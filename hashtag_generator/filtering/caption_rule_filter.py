import os
import json
from utils.caption_filtering_utils import filter_by_keywords

INPUT_JSON_DIR = "./data/captioned_json"
OUTPUT_JSON_DIR = "./data/captioned_json_filtered"
FAILED_DIR = "./data/captioned_json_failed"

os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)

def filter_caption_files():
    for fname in os.listdir(INPUT_JSON_DIR):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(INPUT_JSON_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        caption = data.get("caption", "")
        if not caption:
            print(f"[⚠️ No Caption] {fname}")
            continue

        keywords, result, reason = filter_by_keywords(caption)

        data["keywords"] = keywords
        data["rule_judgement"] = result
        data["rule_reason"] = reason

        out_path = os.path.join(
            OUTPUT_JSON_DIR if result == "pass" else FAILED_DIR, fname
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[{result.upper()}] {fname} → {reason}")

if __name__ == "__main__":
    filter_caption_files()
