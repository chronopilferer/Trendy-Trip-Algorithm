import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from img2hastag.utils.io import save_result, copy_image
from img2hastag.utils.text import extract_judgement

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

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    judgement = extract_judgement(text, prompt)
    return judgement.lower(), text

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

    json_path = Path(json_dir)
    category = json_path.name
    base_output = Path(output_dir).parent       
    final_base = base_output / category

    json_out    = final_base / "json"
    pass_dir    = final_base / "pass"
    partial_dir = final_base / "partial-pass"
    nonpass_dir = final_base / "non-pass"
    for d in (json_out, pass_dir, partial_dir, nonpass_dir):
        d.mkdir(parents=True, exist_ok=True)

    for json_file in json_path.glob("*.json"):
        rec = json.loads(json_file.read_text(encoding="utf-8"))

        caption1 = rec.get("caption_instructblip", "").strip()
        caption2 = rec.get("caption_llava", "").strip()
        img_path = rec.get("file_path", "")
        img_name = Path(img_path).name

        if rec.get("pass") is False:
            copy_image(img_path, str(final_base), 'non-pass')
            continue
        if skip_if_judged and all(
            f"{response_field_name}_{i}" in rec for i in (1, 2)
        ):
            continue

        if not caption1 or not caption2 or not Path(img_path).is_file():
            print(f"[⚠️ 스킵됨] {json_file.name} - 캡션 또는 이미지 누락")
            continue

        print(f"[처리 중] {json_file.name}")
        try:
            judgement1, response1 = process_judgement(
                caption1, tokenizer, model, device,
                prompt_template, max_new_tokens, temperature, top_p
            )
            judgement2, response2 = process_judgement(
                caption2, tokenizer, model, device,
                prompt_template, max_new_tokens, temperature, top_p
            )

            is_pass1 = any(kw in judgement1 for kw in suitable_keywords)
            is_pass2 = any(kw in judgement2 for kw in suitable_keywords)

            rec.update({
                "judgement_1": judgement1,
                f"{response_field_name}_1": response1,
                "pass_1": is_pass1,
                "judgement_2": judgement2,
                f"{response_field_name}_2": response2,
                "pass_2": is_pass2,
            })
            if is_pass1 and is_pass2:
                pass_type = "pass"
            elif is_pass1 or is_pass2:
                pass_type = "partial-pass"
            else:
                pass_type = "non-pass"
            rec["pass_type"] = pass_type

            out_json = json_out / json_file.name
            save_result(rec, str(out_json))
            print(f"[JSON 저장됨] {out_json}")

            dest_dir = final_base / pass_type
            copy_image(img_path, str(final_base), pass_type)
            print(f"[이미지 저장됨] {img_name} → {dest_dir}\n")

        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ [OOM 오류] {json_file.name} - {e}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ [에러] {json_file.name} - {e}")
