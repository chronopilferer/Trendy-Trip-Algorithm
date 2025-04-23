import os
import json
from typing import List, Set, Tuple

from utils.text import (
    download_nltk_resources,
    init_nlp_tools,
    extract_keywords_from_caption
)
from utils.file_io import save_result, copy_image

def filter_by_keywords(
    keywords: List[str],
    food_keywords: Set[str],
    emotion_keywords: Set[str]
) -> Tuple[List[str], str, str]:
    has_food = any(k in food_keywords for k in keywords)
    has_emotion = any(k in emotion_keywords for k in keywords)

    if has_food and not has_emotion:
        return keywords, "fail", "food_only"
    if has_emotion and not has_food:
        return keywords, "fail", "emotion_only"
    return keywords, "pass", "sufficient_context"

def process_caption_rule_filtering(
    json_dir: str,
    output_dir: str,
    food_keywords: Set[str],
    emotion_keywords: Set[str]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for label in ['pass', 'non-pass']:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    download_nltk_resources()
    lemmatizer, stop_words = init_nlp_tools()

    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]

    for fname in json_files:
        json_path = os.path.join(json_dir, fname)
        with open(json_path, 'r', encoding='utf-8') as jf:
            rec = json.load(jf)

        if rec.get('pass') is False:
            continue

        caption = rec.get('caption', '').strip()
        img_path = rec.get('filepath', '').strip()

        if not caption:
            print(f"[⚠️ No Caption] {fname}")
            continue
        if not img_path or not os.path.isfile(img_path):
            print(f"[⚠️ 이미지 없음] {fname}")
            continue

        keywords = extract_keywords_from_caption(caption, lemmatizer, stop_words)
        keywords, result, reason = filter_by_keywords(
            keywords,
            food_keywords,
            emotion_keywords
        )

        rec.update({
            'keywords': keywords,
            'rule_judgement': result,
            'rule_reason': reason,
            'pass': (result == 'pass')
        })

        judgement_dir = 'pass' if rec['pass'] else 'non-pass'
        target_dir = os.path.join(output_dir, judgement_dir)
        os.makedirs(target_dir, exist_ok=True)

        output_json_path = os.path.join(target_dir, fname)
        save_result(rec, output_json_path)
        print(f"[JSON 저장됨] {output_json_path}")

        copy_image(img_path, target_dir, judgement_dir)
        print(f"[이미지 복사됨] {fname} → {target_dir}")

        print(f"[결과] rule_judgement: {result}, rule_reason: {reason}\n")
