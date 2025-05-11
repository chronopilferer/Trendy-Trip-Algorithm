
def extract_judgement(response: str, prompt: str) -> str:
    response_clean = response.replace(prompt, '').strip().lower()

    if "unsuitable" in response_clean:
        return "non-pass"
    elif "suitable" in response_clean:
        return "pass"
    else:
        return "unknown"