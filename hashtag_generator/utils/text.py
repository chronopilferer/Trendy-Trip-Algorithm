import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Set

def download_nltk_resources():
    resources = [
        'punkt', 
        'averaged_perceptron_tagger', 
        'wordnet', 
        'stopwords'
    ]
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}')
        except LookupError:
            nltk.download(r)

def init_nlp_tools():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

def extract_keywords_from_object_phrase(
    phrase: str,
    lemmatizer: WordNetLemmatizer,
    stop_words: Set[str]
) -> List[str]:
    tokens = nltk.word_tokenize(phrase.lower())
    return [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in stop_words
    ]

def extract_keywords_from_caption(
    caption: str,
    lemmatizer: WordNetLemmatizer,
    stop_words: Set[str]
) -> List[str]:
    # 콤마로 구분된 구절별로 키워드 추출
    objects = [obj.strip() for obj in caption.lower().split(",")]
    all_keywords: List[str] = []
    for obj in objects:
        all_keywords.extend(
            extract_keywords_from_object_phrase(obj, lemmatizer, stop_words)
        )
    return list(set(all_keywords))

def extract_judgement(response: str, prompt: str) -> str:
    response_clean = response.replace(prompt, '').strip().lower()

    if "suitable" in response_clean:
        return "pass"
    elif "unsuitable" in response_clean:
        return "non-pass"
    else:
        return "unknown"