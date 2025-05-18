import json
import torch
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from img2hastag.utils.io import save_result

logger = logging.getLogger(__name__)

def load_filtering_model(
    model_id: str,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16
):
    print(f"[모델 로딩] {model_id}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    return tokenizer, model

import re

def process_llm_score(
    caption: str,
    tokenizer,
    model,
    device: str,
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> int:
    prompt = prompt_template.replace("{caption}", caption)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            early_stopping=True,
            do_sample=True
        )

    score_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    logging.info(f'score_response: {score_response}')

    # 숫자 추출 (정규식 기반)
    match = re.search(r'\b([1-9]|10)\b', score_response)
    if match:
        return int(match.group(1))
    else:
        logging.warning(f"❗ 점수 파싱 실패, 기본값 0 반환: '{score_response}'")
        return 0

def process_llm_filtering(
    json_dir: str,
    output_dir: str,
    tokenizer,
    model,
    device: str,
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    skip_if_judged: bool = True,
    response_field_name: str = "LLM_response"
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    json_path = Path(json_dir)
    output_path = Path(output_dir)
    
    # score 저장 위치
    json_out = output_path
    pass_dir = output_path.parent / "pass"
    partial_dir = output_path.parent / "partial-pass"
    nonpass_dir = output_path.parent / "non-pass"
    
    for d in (json_out, pass_dir, partial_dir, nonpass_dir):
        d.mkdir(parents=True, exist_ok=True)

    for json_file in json_path.glob("*.json"):
        if json_file.stat().st_size == 0:
            logging.info(f"❌ [스킵됨] 파일이 비어 있음: {json_file.name}")
            continue

        try:
            rec = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logging.info(f"❌ [스킵됨] JSON 파싱 실패: {json_file.name} - {e}")
            continue

        caption1 = rec.get("caption_instructblip", "").strip()
        caption2 = rec.get("caption_llava", "").strip()
        img_path = rec.get("file_path", "")
        img_name = Path(img_path).name

        if rec.get("pass_type") != "pass":
            logging.info(f"❌ [스킵됨] pass_type이 'pass'가 아님: {json_file.name}")
            continue

        # if "llm_score_1" in rec and "llm_score_2" in rec:
        #     logging.info(f"❌ [스킵됨] 점수 이미 존재: {json_file.name}")
        #     continue

        out_json = json_out / json_file.name

        if not caption1 or not caption2 or not Path(img_path).is_file():
            logging.info(f"[⚠️ 스킵됨] {json_file.name} - 캡션 또는 이미지 누락")
            continue

        logging.info(f"[처리 중] {json_file.name}")
        try:
            # LLM 점수 계산
            score1 = process_llm_score(
                caption1, tokenizer, model, device,
                prompt_template, max_new_tokens, temperature, top_p
            )
            score2 = process_llm_score(
                caption2, tokenizer, model, device,
                prompt_template, max_new_tokens, temperature, top_p
            )

            # JSON에 점수 추가
            rec["llm_score_1"] = score1
            rec["llm_score_2"] = score2

            logging.info(f'score-1: {score1}')
            logging.info(f'score-2: {score2}')

            # JSON 파일 저장
            save_result(rec, str(out_json))
            logging.info(f"[JSON 저장됨] {out_json}")

            exit()

        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"❌ [OOM 오류] {json_file.name} - {e}")
            torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"❌ [에러] {json_file.name} - {e}")
