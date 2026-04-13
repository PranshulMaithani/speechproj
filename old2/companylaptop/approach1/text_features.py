"""
Text feature extraction from Whisper transcripts.

These features capture the LINGUISTIC signal:
- Cheaters: complex vocabulary, structured sentences, no filler words
- Genuine: common vocabulary, repetition, self-reference, discourse markers, fillers
"""
import re
import math
from collections import Counter


# ── Filler / discourse word lists ─────────────────────────────────────────
FILLER_WORDS = {
    "um", "uh", "uhh", "umm", "hmm", "hm", "er", "erm", "ah", "ahh",
    "like", "basically", "literally", "actually", "right",
}

DISCOURSE_MARKERS = {
    "so", "well", "anyway", "ok", "okay", "yeah", "yep", "sure",
    "i mean", "you know", "you see", "i think", "i guess", "i feel",
    "kind of", "sort of", "i suppose",
}

SELF_REFERENCE_WORDS = {"i", "me", "my", "myself", "mine", "we", "our", "us"}

# Words that are hallmarks of AI-generated / overly formal text
COMPLEX_MARKERS = {
    "comprehensive", "multifaceted", "leverage", "leveraging", "furthermore",
    "additionally", "moreover", "consequently", "nevertheless", "notwithstanding",
    "paramount", "facilitate", "utilize", "utilizing", "encompass", "encompassing",
    "streamline", "synergy", "holistic", "robust", "innovative", "strategic",
    "implement", "implementing", "implementation", "optimize", "optimizing",
    "enhance", "enhancing", "crucial", "pivotal", "significant", "significantly",
    "demonstrate", "demonstrating", "articulate", "articulating", "proficient",
    "meticulous", "meticulously", "adept", "intricate", "intrinsically",
    "delve", "delving", "realm", "landscape", "paradigm", "framework",
    "methodology", "aforementioned", "henceforth", "thereby", "wherein",
    "endeavor", "endeavoring", "culminate", "culminating",
}


def _count_syllables(word: str) -> int:
    """Rough syllable count using vowel-cluster heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    # Count vowel groups
    count = len(re.findall(r'[aeiouy]+', word))
    # Adjust for silent e
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization: lowercase, keep only alpha tokens."""
    return [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 2]


def extract_text_features(text: str, duration_sec: float = None) -> dict:
    """
    Extract linguistic features from a transcript.

    Args:
        text: Full transcript text from Whisper
        duration_sec: Audio duration in seconds (for speaking rate)

    Returns:
        Dictionary of text features
    """
    features = {}
    words = _tokenize(text)
    n_words = len(words)

    # ── Handle empty / very short transcripts ─────────────────────────
    if n_words < 3:
        return {k: 0.0 for k in [
            "word_count", "unique_word_count", "type_token_ratio",
            "avg_word_length", "long_word_rate", "complex_word_rate",
            "complex_marker_rate", "filler_rate", "filler_count",
            "discourse_marker_rate", "self_reference_rate",
            "repetition_rate_bigram", "repetition_rate_trigram",
            "avg_sentence_length", "sentence_length_std", "sentence_count",
            "words_per_minute", "char_per_word",
            "hapax_ratio", "top10_word_concentration",
        ]}

    word_set = set(words)
    word_counter = Counter(words)

    # ── Vocabulary complexity ─────────────────────────────────────────
    features["word_count"] = float(n_words)
    features["unique_word_count"] = float(len(word_set))
    features["type_token_ratio"] = len(word_set) / n_words

    # Average word length (characters)
    features["avg_word_length"] = sum(len(w) for w in words) / n_words
    features["char_per_word"] = features["avg_word_length"]

    # Long words (7+ chars) and complex words (3+ syllables)
    long_words = [w for w in words if len(w) >= 7]
    features["long_word_rate"] = len(long_words) / n_words

    complex_words = [w for w in words if _count_syllables(w) >= 3]
    features["complex_word_rate"] = len(complex_words) / n_words

    # AI/formal language markers
    marker_hits = sum(1 for w in words if w in COMPLEX_MARKERS)
    features["complex_marker_rate"] = marker_hits / n_words

    # Hapax legomena (words appearing exactly once) -- high = more diverse vocab
    hapax = sum(1 for w, c in word_counter.items() if c == 1)
    features["hapax_ratio"] = hapax / n_words

    # Top-10 word concentration -- how much of the text is the 10 most common words
    top10_count = sum(c for _, c in word_counter.most_common(10))
    features["top10_word_concentration"] = top10_count / n_words

    # ── Disfluency / naturalness markers ──────────────────────────────
    # Filler words
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    features["filler_rate"] = filler_count / n_words
    features["filler_count"] = float(filler_count)

    # Discourse markers (some are multi-word, check in raw text)
    text_lower = text.lower()
    dm_count = 0
    for dm in DISCOURSE_MARKERS:
        dm_count += text_lower.count(dm)
    features["discourse_marker_rate"] = dm_count / n_words

    # Self-reference
    self_ref = sum(1 for w in words if w in SELF_REFERENCE_WORDS)
    features["self_reference_rate"] = self_ref / n_words

    # ── Repetition ────────────────────────────────────────────────────
    # Bigram repetition rate
    if n_words >= 4:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(n_words - 1)]
        bigram_counter = Counter(bigrams)
        repeated_bigrams = sum(c - 1 for c in bigram_counter.values() if c > 1)
        features["repetition_rate_bigram"] = repeated_bigrams / len(bigrams)
    else:
        features["repetition_rate_bigram"] = 0.0

    # Trigram repetition
    if n_words >= 6:
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(n_words - 2)]
        trigram_counter = Counter(trigrams)
        repeated_trigrams = sum(c - 1 for c in trigram_counter.values() if c > 1)
        features["repetition_rate_trigram"] = repeated_trigrams / len(trigrams)
    else:
        features["repetition_rate_trigram"] = 0.0

    # ── Sentence structure ────────────────────────────────────────────
    sentences = _split_sentences(text)
    n_sentences = max(len(sentences), 1)
    features["sentence_count"] = float(n_sentences)

    sent_lengths = [len(_tokenize(s)) for s in sentences]
    if sent_lengths:
        features["avg_sentence_length"] = sum(sent_lengths) / len(sent_lengths)
        features["sentence_length_std"] = (
            (sum((l - features["avg_sentence_length"])**2 for l in sent_lengths) / len(sent_lengths)) ** 0.5
        )
    else:
        features["avg_sentence_length"] = 0.0
        features["sentence_length_std"] = 0.0

    # ── Speaking rate ─────────────────────────────────────────────────
    if duration_sec and duration_sec > 0:
        features["words_per_minute"] = (n_words / duration_sec) * 60
    else:
        features["words_per_minute"] = 0.0

    return features
