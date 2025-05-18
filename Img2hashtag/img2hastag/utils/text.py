import re

def clean_response(response_text: str, prompt: str) -> str:
    response_only = response_text[len(prompt):].strip()
    return response_only

def extract_judgement(response_text: str) -> str:
    match = re.search(r"(Suitable|Unsuitable)", response_text, re.IGNORECASE)
    
    if match:
        return match.group(0).lower()  
    else:
        return "unsuitable"  
