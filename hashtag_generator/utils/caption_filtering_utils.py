import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from hashtag_generator.utils.config import load_config

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

_config = load_config()
_filtering_cfg = _config.get("caption_filtering", {})

FOOD_KEYWORDS = set(_filtering_cfg.get("food_keywords", []))
EMOTIONAL_CONTEXT_KEYWORDS = set(_filtering_cfg.get("emotional_context_keywords", []))

def extract_keywords_from_object_phrase(phrase: str) -> list:
    tokens = nltk.word_tokenize(phrase.lower())
    keywords = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return keywords

def extract_keywords_from_caption(caption: str) -> list:
    objects = [obj.strip() for obj in caption.lower().split(",")]
    all_keywords = []
    for obj in objects:
        all_keywords.extend(extract_keywords_from_object_phrase(obj))
    return list(set(all_keywords))  

def filter_by_keywords(keywords: list) -> tuple:
    has_food = any(word for word in keywords)
    has_emotion = any(word for word in keywords)

    if has_food and not has_emotion:
        return "fail", "food_only"
    return "pass", "sufficient_context"

def process_caption_with_rule(caption: str) -> tuple:
    keywords = extract_keywords_from_caption(caption)
    result, reason = filter_by_keywords(keywords)
    return keywords, result, reason