"""Text-feature extraction from transcripts.

Runs on the company laptop where transcripts (containing PII) are allowed to
exist. Only the numeric features it returns are written into gt.csv and
uploaded; the transcript text itself never leaves this machine.

`compute_text_features(transcript: str)` returns a dict keyed by the names
in TEXT_FEATURE_NAMES (all values are floats). Empty / None / whitespace-only
transcripts return all zeros so downstream code never sees NaN here.

Feature set is intentionally small and dependency-free (regex + sets only).
Mirrors the style of the v3 pipeline's handcrafted features but does NOT
require spacy / nltk / GPT-2.
"""

from __future__ import annotations

import re
from typing import Sequence

# Word lists are lowercase. Multi-word phrases use space-separated form and
# are matched as substrings on the lowered text; single words are matched on
# tokenized words.
FILLERS: tuple[str, ...] = (
    "um", "uh", "er", "ah", "hmm", "mm",
    "like", "you know", "i mean", "basically", "literally", "actually", "right",
)
DISCOURSE_MARKERS: tuple[str, ...] = (
    "so", "well", "okay", "ok", "anyway", "anyways",
    "however", "therefore", "moreover", "furthermore", "in fact", "of course",
)
HEDGES: tuple[str, ...] = (
    "maybe", "perhaps", "probably", "possibly",
    "kind of", "sort of", "i think", "i guess", "i suppose",
    "somewhat", "rather", "a bit", "a little",
)
SELF_REF: tuple[str, ...] = ("i", "me", "my", "myself", "mine")
FORMAL_TRANSITIONS: tuple[str, ...] = (
    "furthermore", "moreover", "additionally", "consequently",
    "nevertheless", "subsequently", "henceforth", "thereby", "wherein",
    "in conclusion", "in summary", "as a result",
)

TEXT_FEATURE_NAMES: list[str] = [
    "feat_word_count",
    "feat_unique_words",
    "feat_ttr",
    "feat_mean_word_length",
    "feat_long_word_rate",
    "feat_complex_word_rate",
    "feat_filler_count",
    "feat_filler_rate",
    "feat_discourse_count",
    "feat_discourse_rate",
    "feat_hedge_count",
    "feat_hedge_rate",
    "feat_self_ref_count",
    "feat_self_ref_rate",
    "feat_formal_count",
    "feat_formal_rate",
    "feat_sentence_count",
    "feat_avg_words_per_sentence",
]

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+")


def _zero_features() -> dict[str, float]:
    return {name: 0.0 for name in TEXT_FEATURE_NAMES}


def _count_phrases(lowered: str, words: list[str], phrases: Sequence[str]) -> int:
    """Count phrase occurrences. Multi-word phrases count substring hits on
    the lowered text; single-word phrases count exact token matches so that
    e.g. "I" doesn't accidentally match "is"."""
    n = 0
    for ph in phrases:
        if " " in ph:
            n += lowered.count(ph)
        else:
            n += sum(1 for w in words if w == ph)
    return n


def compute_text_features(transcript) -> dict[str, float]:
    if not isinstance(transcript, str) or not transcript.strip():
        return _zero_features()

    lowered = transcript.lower()
    words = _WORD_RE.findall(lowered)
    n_words = len(words)
    if n_words == 0:
        return _zero_features()

    feats: dict[str, float] = {}
    feats["feat_word_count"] = float(n_words)
    feats["feat_unique_words"] = float(len(set(words)))
    feats["feat_ttr"] = feats["feat_unique_words"] / n_words
    feats["feat_mean_word_length"] = float(sum(len(w) for w in words)) / n_words
    feats["feat_long_word_rate"] = sum(1 for w in words if len(w) >= 7) / n_words
    feats["feat_complex_word_rate"] = sum(1 for w in words if len(w) >= 9) / n_words

    f_filler = _count_phrases(lowered, words, FILLERS)
    f_disc = _count_phrases(lowered, words, DISCOURSE_MARKERS)
    f_hedge = _count_phrases(lowered, words, HEDGES)
    f_self = _count_phrases(lowered, words, SELF_REF)
    f_formal = _count_phrases(lowered, words, FORMAL_TRANSITIONS)

    feats["feat_filler_count"] = float(f_filler)
    feats["feat_filler_rate"] = f_filler / n_words
    feats["feat_discourse_count"] = float(f_disc)
    feats["feat_discourse_rate"] = f_disc / n_words
    feats["feat_hedge_count"] = float(f_hedge)
    feats["feat_hedge_rate"] = f_hedge / n_words
    feats["feat_self_ref_count"] = float(f_self)
    feats["feat_self_ref_rate"] = f_self / n_words
    feats["feat_formal_count"] = float(f_formal)
    feats["feat_formal_rate"] = f_formal / n_words

    sentences = [s for s in _SENT_RE.split(transcript) if s.strip()]
    n_sents = max(len(sentences), 1)
    feats["feat_sentence_count"] = float(n_sents)
    feats["feat_avg_words_per_sentence"] = float(n_words / n_sents)

    return feats
