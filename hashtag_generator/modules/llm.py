import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.file_io import save_result, copy_image
from utils.text import extract_judgement

def load_filtering_model(
    model_id: str,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16
):
    print(f"[모델 로딩] {model_id}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
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

def process_judgement(
    caption: str,
    tokenizer,
    model,
    device: str,
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple:
    # 프롬프트에 캡션 삽입
    prompt = prompt_template.replace("{caption}", caption.strip())
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            early_stopping=True,
            do_sample=False
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    judgement = extract_judgement(text, prompt)
    return judgement.lower(), text.strip()

def process_llm_filtering(
    json_dir: str,
    output_dir: str,
    model_id: str,
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    suitable_keywords: list = ["suitable"],
    skip_if_judged: bool = True,
    response_field_name: str = "LLM_response"
) -> None:
    tokenizer, model = load_filtering_model(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
        torch_dtype=torch_dtype
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    for label in ['pass', 'non-pass']:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]

    for fname in json_files:
        path = os.path.join(json_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            rec = json.load(f)

        # 이전 필터링에서 실패한 항목 무시
        if rec.get("pass") is False:
            continue

        if skip_if_judged and response_field_name in rec:
            continue

        caption = rec.get("caption", "")
        img_path = rec.get("filepath", "")
        img_name = rec.get("filename", fname.replace(".json", ".jpg"))

        if not caption or not img_path or not os.path.isfile(img_path):
            print(f"[⚠️ 스킵됨] {fname} - 누락된 caption/이미지")
            continue

        print(f"[처리 중] {fname}")
        try:
            judgement, response = process_judgement(
                caption=caption,
                tokenizer=tokenizer,
                model=model,
                device=device,
                prompt_template=prompt_template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            is_pass = any(keyword in judgement for keyword in suitable_keywords)

            rec.update({
                "judgement": judgement,
                response_field_name: response,
                "pass": is_pass
            })

            label = 'pass' if is_pass else 'non-pass'
            target_dir = os.path.join(output_dir, label)
            os.makedirs(target_dir, exist_ok=True)

            out_json = os.path.join(target_dir, fname)
            save_result(rec, out_json)
            print(f"[JSON 저장됨] {out_json}")

            copy_image(img_path, target_dir, label)
            print(f"[이미지 저장됨] {img_name} → {target_dir}\n")

        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ [OOM 오류] {fname} - {e}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ [에러] {fname} - {e}")
