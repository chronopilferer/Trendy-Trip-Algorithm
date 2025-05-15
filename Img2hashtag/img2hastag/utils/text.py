
def extract_judgement(text: str, prompt: str) -> str:
    answer = text.replace(prompt, "").strip().lower()
    if "suitable" in answer:
        return "suitable"
    elif "unsuitable" in answer:
        return "unsuitable"
    else:
        return answer  