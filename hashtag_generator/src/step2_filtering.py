import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_filtering_model():
    print("[모델 로딩] Gemma-3 12B IT")
    model_id = "google/gemma-3-12b-it"
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

def extract_judgement(response: str, prompt: str) -> str:
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    for word in response.split():
        word_lower = word.lower().strip(".,!?")
        if word_lower == "suitable":
            return "Suitable"
        elif word_lower == "unsuitable":
            return "Unsuitable"
    return "Unknown"

def filter_caption(caption, tokenizer, model, device):
    prompt = (
        f'The following caption is intended for use in generating social media hashtags.\n'
        f'Caption: "{caption.strip()}"\n'
        f'Does this caption contain enough descriptive and relevant content to generate effective hashtags?\n'
        f'Answer with one word: Suitable or Unsuitable.'
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    outputs = model.generate(**inputs, max_new_tokens=30)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    judgement = extract_judgement(response, prompt)
    
    return judgement, response

def process_json_files(input_folder, output_folder, tokenizer, model, device):
    os.makedirs(output_folder, exist_ok=True)
    json_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".json")]
    
    for json_file in json_files:
        json_path = os.path.join(input_folder, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        judgement = data.get("judgement", "").strip()

        if judgement is not None:
            continue

        caption = data.get("caption", "").strip()
        if caption:
            judgement, filter_response = filter_caption(caption, tokenizer, model, device)
        else:
            judgement, filter_response = "No Caption", "No caption text provided."
        data["judgement"] = judgement
        data["filter_response"] = filter_response
        
        output_path = os.path.join(output_folder, json_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"파일 처리 완료: {json_file} - 판단: {judgement}")

if __name__ == "__main__":
    input_folder = "./captions"
    output_folder = "./captions"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_filtering_model()
    process_json_files(input_folder, output_folder, tokenizer, model, device)
