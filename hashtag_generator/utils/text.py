import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Set

def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'wordnet']
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}')
        except LookupError:
            nltk.download(r)

def init_nlp_tools():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

def extract_keywords_from_object_phrase(phrase: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> List[str]:
    tokens = nltk.word_tokenize(phrase.lower())
    return [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in stop_words
    ]

def extract_keywords_from_caption(caption: str, lemmatizer: WordNetLemmatizer, stop_words: Set[str]) -> List[str]:
    objects = [obj.strip() for obj in caption.lower().split(",")]
    all_keywords = []
    for obj in objects:
        all_keywords.extend(
            extract_keywords_from_object_phrase(obj, lemmatizer, stop_words)
        )
    return list(set(all_keywords))

def extract_judgement(
    response: str,
    prompt: str,
    suitable_keywords: list,
    unsuitable_keywords: list
) -> str:
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    for word in response.split():
        w = word.lower().strip('.,!?"')
        if w in suitable_keywords:
            return "Suitable"
        if w in unsuitable_keywords:
            return "Unsuitable"
    return "Unknown"

def filter_by_keywords(keywords: list) -> tuple:
    has_food = any(word for word in keywords)
    has_emotion = any(word for word in keywords)

    if has_food and not has_emotion:
        return "fail", "food_only"
    return "pass", "sufficient_context"