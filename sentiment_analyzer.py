import re
import spacy
from typing import List, Dict

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
    nlp = None

# Excitement indicators with lemma forms
EXCITEMENT_LEMMAS = {
    "goal", "score", "brilliant", "amazing", "incredible", "fantastic",
    "unbelievable", "wow", "beautiful", "stunning", "magnificent",
    "sensational", "spectacular", "wonderful", "superb", "perfect",
    "genius", "magic", "masterclass", "clinical", "unstoppable"
}

# Negative/tension indicators
TENSION_LEMMAS = {
    "foul", "card", "injury", "hurt", "miss", "save", "block",
    "dangerous", "close", "almost", "nearly", "offside", "controversial"
}

# Intensifier words that boost sentiment
INTENSIFIERS = {
    "very", "so", "really", "absolutely", "completely", "totally",
    "incredibly", "extremely", "truly", "just"
}

# Negation words that flip sentiment
NEGATIONS = {"not", "no", "never", "none", "nothing", "neither", "nobody"}

INTENSITY_PATTERNS = {
    "high": [
        r"!{2,}",                    # Multiple exclamation marks
        r"[A-Z]{3,}",                # Caps lock words
        r"\b(GOAL|YES|WOW|OH)\b",    # Shouted words
        r"what a (goal|strike|save|finish|pass|hit)",
        r"(brilliant|amazing|incredible|unbelievable|sensational)"
    ],
    "medium": [
        r"!",                        # Single exclamation
        r"(good|nice|well done|great)",
        r"(chance|shot|opportunity|attack)"
    ],
    "low": [
        r"(pass|possession|back|sideways)",
        r"(waiting|looking|patient|calm)"
    ]
}


def analyze_with_spacy(text: str) -> Dict:
    """Use spaCy for linguistic analysis of the text."""
    if nlp is None:
        return {"excitement_score": 0, "tension_score": 0, "has_intensifier": False, "has_negation": False}
    
    doc = nlp(text)
    
    excitement_count = 0
    tension_count = 0
    intensifier_count = 0
    has_negation = False
    
    # Analyze each token
    for token in doc:
        lemma = token.lemma_.lower()
        
        # Check for excitement words
        if lemma in EXCITEMENT_LEMMAS:
            excitement_count += 1
        
        # Check for tension words
        if lemma in TENSION_LEMMAS:
            tension_count += 1
        
        # Check for intensifiers
        if lemma in INTENSIFIERS:
            intensifier_count += 1
        
        # Check for negations
        if lemma in NEGATIONS:
            has_negation = True
    
    # Analyze adjectives and adverbs for additional sentiment
    adjectives = [token for token in doc if token.pos_ in ["ADJ", "ADV"]]
    strong_adj_count = sum(1 for adj in adjectives if adj.lemma_.lower() in EXCITEMENT_LEMMAS)
    
    # Calculate scores
    word_count = len([t for t in doc if not t.is_punct and not t.is_space])
    
    excitement_score = (excitement_count + strong_adj_count * 0.5) / max(word_count, 1)
    tension_score = tension_count / max(word_count, 1)
    
    return {
        "excitement_score": min(excitement_score * 3, 1.0),  # Normalize
        "tension_score": min(tension_score * 3, 1.0),
        "has_intensifier": intensifier_count > 0,
        "has_negation": has_negation,
        "adjective_count": len(adjectives),
        "word_count": word_count
    }


def calculate_intensity(text: str) -> float:
    """Calculate emotional intensity score (0-1) using spaCy and patterns."""
    text_lower = text.lower()
    score = 0.0
    
    # spaCy-based analysis
    spacy_analysis = analyze_with_spacy(text)
    score += spacy_analysis["excitement_score"] * 0.4
    score += spacy_analysis["tension_score"] * 0.2
    
    # Intensifier boost
    if spacy_analysis["has_intensifier"]:
        score += 0.1
    
    # Check for high intensity patterns
    for pattern in INTENSITY_PATTERNS["high"]:
        if re.search(pattern, text, re.IGNORECASE):
            score += 0.2
    
    # Check for medium intensity patterns
    for pattern in INTENSITY_PATTERNS["medium"]:
        if re.search(pattern, text, re.IGNORECASE):
            score += 0.08
    
    # Exclamation marks boost
    exclamation_count = text.count("!")
    score += min(exclamation_count * 0.08, 0.25)
    
    # Question marks (uncertainty/anticipation)
    question_count = text.count("?")
    score += min(question_count * 0.05, 0.1)
    
    # Caps lock ratio (shouting)
    caps_chars = sum(1 for c in text if c.isupper())
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars > 0:
        caps_ratio = caps_chars / alpha_chars
        if caps_ratio > 0.3:
            score += 0.15
    
    # Cap at 1.0
    return min(score, 1.0)


def classify_moment(intensity: float) -> str:
    """Classify moment as exciting, moderate, or calm."""
    if intensity >= 0.6:
        return "exciting"
    elif intensity >= 0.3:
        return "moderate"
    else:
        return "calm"


def get_sentiment_label(intensity: float, tension_score: float = 0) -> str:
    """Get a more descriptive sentiment label."""
    if intensity >= 0.7:
        return "very_exciting"
    elif intensity >= 0.5:
        if tension_score > 0.3:
            return "tense"
        return "exciting"
    elif intensity >= 0.3:
        return "moderate"
    else:
        return "calm"


def calculate_intensity_with_volume(text: str, volume_score: float = 0.0) -> float:
    """Calculate emotional intensity score (0-1) using text analysis + audio volume."""
    # Base text intensity
    text_intensity = calculate_intensity(text)
    
    # Weight: 60% text, 40% audio volume
    combined_intensity = (text_intensity * 0.6) + (volume_score * 0.4)
    
    # If both are high, boost the score
    if text_intensity > 0.5 and volume_score > 0.5:
        combined_intensity = min(combined_intensity + 0.1, 1.0)
    
    return round(combined_intensity, 2)


def analyze_sentiment(segments: List[Dict], segments_with_volume: List[Dict] = None) -> List[Dict]:
    """Analyze sentiment/intensity for each segment using spaCy and optional audio volume."""
    results = []
    
    # Create volume lookup if available
    volume_lookup = {}
    if segments_with_volume:
        for seg in segments_with_volume:
            time_key = round(seg.get("start", 0), 1)
            volume_lookup[time_key] = seg.get("volume_score", 0)
    
    for segment in segments:
        text = segment.get("text", "")
        start_time = segment.get("start", 0)
        spacy_analysis = analyze_with_spacy(text)
        
        # Get volume score if available
        volume_score = volume_lookup.get(round(start_time, 1), 0)
        
        # Calculate intensity with or without volume
        if volume_score > 0:
            intensity = calculate_intensity_with_volume(text, volume_score)
        else:
            intensity = calculate_intensity(text)
        
        result = {
            "time": start_time,
            "text": text.strip(),
            "intensity": intensity,
            "mood": classify_moment(intensity),
            "sentiment": get_sentiment_label(intensity, spacy_analysis["tension_score"]),
            "linguistic_features": {
                "excitement_score": round(spacy_analysis["excitement_score"], 2),
                "tension_score": round(spacy_analysis["tension_score"], 2),
                "has_intensifier": spacy_analysis["has_intensifier"],
                "word_count": spacy_analysis["word_count"]
            }
        }
        
        # Add volume info if available
        if volume_score > 0:
            result["audio_features"] = {
                "volume_score": volume_score
            }
        
        results.append(result)
    
    return results


def get_exciting_moments(analyzed: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """Get moments above intensity threshold."""
    return [m for m in analyzed if m["intensity"] >= threshold]


def get_peak_moments(analyzed: List[Dict], top_n: int = 5) -> List[Dict]:
    """Get the top N most intense moments."""
    sorted_moments = sorted(analyzed, key=lambda x: x["intensity"], reverse=True)
    return sorted_moments[:top_n]


def get_intensity_summary(analyzed: List[Dict]) -> Dict:
    """Get comprehensive summary of intensity distribution."""
    moods = {"exciting": 0, "moderate": 0, "calm": 0}
    sentiments = {"very_exciting": 0, "exciting": 0, "tense": 0, "moderate": 0, "calm": 0}
    
    for m in analyzed:
        moods[m["mood"]] += 1
        sentiments[m.get("sentiment", "calm")] += 1
    
    total = len(analyzed)
    avg_intensity = sum(m["intensity"] for m in analyzed) / total if total > 0 else 0
    
    # Find peak intensity
    peak = max(analyzed, key=lambda x: x["intensity"]) if analyzed else None
    
    return {
        "average_intensity": round(avg_intensity, 2),
        "peak_intensity": round(peak["intensity"], 2) if peak else 0,
        "peak_moment_time": peak["time"] if peak else 0,
        "mood_distribution": moods,
        "sentiment_distribution": sentiments,
        "total_segments": total,
        "exciting_percentage": round(moods["exciting"] / total * 100, 1) if total > 0 else 0
    }
