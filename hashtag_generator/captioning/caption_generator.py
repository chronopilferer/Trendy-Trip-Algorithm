from PIL import Image

def generate_caption(image_path, processor, model, prompt, device):
    image = Image.open(image_path).convert("RGB")
    modified_prompt = prompt.strip() + "\nAnswer:"
    inputs = processor(images=image, text=modified_prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    outputs = model.generate(**inputs, max_new_tokens=256)
    caption = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    if caption.startswith(modified_prompt):
        caption = caption[len(modified_prompt):].strip()

    return caption
