import spacy
import re


# Load English model
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text: str) -> dict:
    """
    Apply NLP preprocessing to text using spaCy.
    Returns tokenized, cleaned, and lemmatized text.
    """
    # Noise removal - remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Process with spaCy
    doc = nlp(text)
    
    # Sentence segmentation
    sentences = [sent.text for sent in doc.sents]
    
    # Tokenization
    tokens = [token.text for token in doc]
    
    # Stopword filtering + Lemmatization
    clean_tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    
    return {
        "original": text,
        "sentences": sentences,
        "tokens": tokens,
        "clean_tokens": clean_tokens,
        "clean_text": " ".join(clean_tokens)
    }
