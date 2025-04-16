from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_filtering_model(model_id: str):
    print(f"[모델 로딩] {model_id}")
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model

def extract_judgement(response: str, prompt: str, config: dict) -> str:
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    for word in response.split():
        word_lower = word.lower().strip(".,!?\"")
        if word_lower in config.get("suitable_keywords", []):
            return "Suitable"
        elif word_lower in config.get("unsuitable_keywords", []):
            return "Unsuitable"
    return "Unknown"

def filter_caption(caption: str, tokenizer, model, device: str, config: dict) -> tuple:
    prompt = config["prompt_template"].replace("{caption}", caption.strip())
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.get("max_new_tokens", 30),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    judgement = extract_judgement(response, prompt, config)
    return judgement, response

def copy_image_by_judgement(img_path, filtered_base_dir, judgement):
    if not img_path or not os.path.exists(img_path):
        print(f"[이미지 없음] {img_path}")
        return

    file_name = os.path.basename(img_path)
    target_dir = os.path.join(filtered_base_dir, judgement)
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(img_path, os.path.join(target_dir, file_name))
    print(f"[이미지 복사됨] {img_path} → {target_dir}")