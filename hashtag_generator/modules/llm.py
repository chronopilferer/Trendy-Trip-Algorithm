import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.file_io import save_result, copy_image
from utils.text import extract_judgement

def load_filtering_model(
    model_id: str,
    load_in_8bit: bool = True,
    llm_int8_enable_fp32_cpu_offload: bool = True,
    device_map: str = "auto",
    torch_dtype="auto"
):
    print(f"[모델 로딩] {model_id}")
    quant_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch_dtype
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
    suitable_keywords: list,
    unsuitable_keywords: list
) -> tuple:
    prompt = prompt_template.replace("{caption}", caption.strip())
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    judgement = extract_judgement(text, prompt, suitable_keywords, unsuitable_keywords)
    return judgement, text

def process_llm_filtering(
    json_dir: str,
    output_dir: str,
    model_id: str,
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    suitable_keywords: list,
    unsuitable_keywords: list,
    load_in_8bit: bool = True,
    llm_int8_enable_fp32_cpu_offload: bool = True,
    device_map: str = "auto",
    torch_dtype: str = "auto"
) -> None:
    tokenizer, model = load_filtering_model(
        model_id,
        load_in_8bit=load_in_8bit,
        llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
        device_map=device_map,
        torch_dtype=torch_dtype
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    for label in ['pass', 'non-pass']:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]

    for fname in json_files:
        json_path = os.path.join(json_dir, fname)

        with open(json_path, "r", encoding="utf-8") as jf:
            rec = json.load(jf)

        if rec.get("pass") is False:
            continue

        caption = rec.get("caption", "")
        img_path = rec.get("filepath", "")
        img_name = rec.get("filename", fname.replace(".json", ".jpg"))

        if not caption:
            print(f"[⚠️ No Caption] {fname}")
            continue
        if not img_path or not os.path.isfile(img_path):
            print(f"[⚠️ 이미지 없음] {fname}")
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
                suitable_keywords=suitable_keywords,
                unsuitable_keywords=unsuitable_keywords
            )

            rec.update({
                "judgement": judgement,
                "LLM_response": response,
                "pass": bool(judgement == "pass")
            })

            judgement_dir = 'pass' if rec["pass"] else 'non-pass'
            json_target_dir = os.path.join(output_dir, judgement_dir)
            img_target_dir = os.path.join(output_dir, judgement_dir)

            os.makedirs(json_target_dir, exist_ok=True)
            os.makedirs(img_target_dir, exist_ok=True)

            out_json_path = os.path.join(json_target_dir, fname)
            save_result(rec, out_json_path)
            print(f"[JSON 저장됨] {out_json_path}")

            copy_image(img_path, img_target_dir, judgement_dir)
            print(f"[이미지 복사됨] {img_name} → {img_target_dir}")

            print(f"[결과] judgement: {judgement}\n")

        except Exception as e:
            print(f"❌ [에러] {fname} - {str(e)}")