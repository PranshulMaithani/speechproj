# Feature Documentation — 41 Features for Speech Cheating Detection

This document describes all 41 features used in the cheating detection pipeline, organized into three groups: Text (20), Pause (13), and Prosodic (8). Each entry includes what the feature measures, how it's computed, and why it's relevant to detecting scripted/read speech vs genuine spontaneous speech.

**Source code:** `extract_features_company.py`

---

## Text Features (20)

These features are extracted from the transcript text using spaCy (`en_core_web_sm`) for tokenization, POS tagging, and sentence segmentation.

### 1. filler_rate
- **What:** Ratio of filler words to total words
- **Formula:** `filler_count / n_words`
- **Filler set:** {um, uh, uh-huh, uhm, umm, hmm, hm, er, ah, ehm, mhm}
- **Why:** Spontaneous speakers produce fillers while planning speech in real-time. Scripted/read speech eliminates them since the text is pre-written. One of the strongest single indicators.
- **Sources:** Clark & Fox Tree (2002) — "Using uh and um in spontaneous speaking", *Cognition*. Showed fillers are produced during lexical retrieval delays, not randomly.

### 2. filler_count
- **What:** Absolute count of filler words
- **Formula:** Count of words matching FILLERS set
- **Why:** Raw count complements the rate — a long genuine response with many fillers behaves differently than a short one with few.

### 3. repetition_rate
- **What:** Proportion of repeated bigrams (word pairs)
- **Formula:** `sum(excess bigram occurrences) / total_bigrams`
- **Why:** Genuine speakers repeat phrases naturally ("I think... I think that's...") due to planning difficulties. Scripts avoid redundancy. 
- **Sources:** Shriberg (1994) — "Preliminaries to a theory of speech disfluencies", PhD thesis, UC Berkeley. Established repetitions as a core disfluency type.

### 4. repair_rate
- **What:** Frequency of self-corrections per sentence
- **Formula:** `repair_count / n_sentences`
- **Repair markers:** ["i mean", "no wait", "sorry i", "actually no", "wait no", "no no"]
- **Why:** Genuine speakers correct themselves mid-utterance. Readers of pre-written text don't make spontaneous errors to correct.
- **Sources:** Levelt (1983) — "Monitoring and self-repair in speech", *Cognition*. Described the self-monitoring mechanism that produces repairs in natural speech.

### 5. ttr (Type-Token Ratio)
- **What:** Lexical diversity — ratio of unique words to total words
- **Formula:** `n_unique_words / n_words`
- **Why:** Scripted or AI-generated text tends to use more diverse vocabulary (higher TTR). Spontaneous speakers reuse common words and have lower TTR.
- **Sources:** Templin (1957) — original TTR metric. Johnson (1944) — early use in language analysis. Known limitation: TTR decreases with text length.

### 6. mattr (Moving Average Type-Token Ratio)
- **What:** Average TTR computed over sliding 50-word windows
- **Formula:** `mean(TTR for each 50-word window)`
- **Why:** Addresses TTR's sensitivity to text length. More stable and comparable across different response durations.
- **Sources:** Covington & McFall (2010) — "Cutting the Gordian knot: The moving-average type-token ratio (MATTR)", *Journal of Quantitative Linguistics*.

### 7. complex_word_rate
- **What:** Proportion of words with 3+ syllables
- **Formula:** `count(words with >= 3 syllables) / n_words`
- **Syllable counting:** Vowel cluster heuristic with silent-e adjustment
- **Why:** Scripted/AI-generated text uses more complex vocabulary. Spontaneous speech defaults to simpler words under time pressure.
- **Sources:** Gunning (1952) — FOG readability index uses 3+ syllable words as complexity indicator. Dale & Chall (1948) — word difficulty and readability.

### 8. avg_word_length
- **What:** Mean character length of all words
- **Formula:** `mean(len(word) for word in words)`
- **Why:** Correlates with formality and preparation. Scripted responses use longer, more formal vocabulary.

### 9. n_words
- **What:** Total word count (alphabetic tokens only)
- **Why:** Response length baseline. Cheaters reading prepared answers may produce longer, more complete responses. Also needed for normalization.

### 10. n_unique_words
- **What:** Count of distinct words (vocabulary size)
- **Why:** Raw vocabulary size, used alongside TTR. Large vocabulary in a short response is suspicious.

### 11. avg_sentence_length
- **What:** Mean number of words per sentence
- **Formula:** `mean(words_per_sentence)`
- **Why:** Scripted text produces longer, well-formed sentences. Spontaneous speech has shorter, fragmented utterances.
- **Sources:** Chafe (1982) — "Integration and involvement in speaking, writing, and oral literature", in *Spoken and Written Language*. Documented structural differences between planned and unplanned speech.

### 12. std_sentence_length
- **What:** Standard deviation of words per sentence
- **Why:** Genuine speech has highly variable sentence lengths (interruptions, restarts, afterthoughts). Scripts are more uniform.

### 13. fragment_rate
- **What:** Proportion of sentence fragments (sentences < 4 words)
- **Formula:** `count(sentences with < 4 words) / n_sentences`
- **Why:** Genuine speech produces many fragments ("Yeah", "So...", "I don't know"). Scripts are composed of complete sentences.
- **Sources:** Biber (1988) — "Variation across speech and writing", Cambridge University Press. Identified fragmentation as a key spoken-language marker.

### 14. n_sentences
- **What:** Total sentence count (spaCy segmentation)
- **Why:** Baseline for normalization. Also, very few sentences in a long audio suggests run-on reading.

### 15. self_ref_rate
- **What:** Frequency of self-referential pronouns per word
- **Formula:** `count(self_ref_words) / n_words`
- **Self-reference set:** {i, me, my, myself, mine, i'm, i've, i'd, i'll}
- **Why:** Genuine responses to personal questions reference the self frequently. Scripts/AI answers are more impersonal and generic.
- **Sources:** Newman et al. (2003) — "Lying words: Predicting deception from linguistic styles", *Personality and Social Psychology Bulletin*. Found self-reference patterns differ between truthful and deceptive speech.

### 16. discourse_marker_rate
- **What:** Frequency of discourse markers per sentence
- **Formula:** `count(markers) / n_sentences`
- **Markers:** {you know, i mean, like, basically, actually, so, well, right, okay, oh, anyway, honestly}
- **Why:** Discourse markers are hallmarks of spontaneous conversation — they manage conversational flow. Scripts don't need them.
- **Sources:** Schiffrin (1987) — "Discourse Markers", Cambridge University Press. Fox Tree (2010) — discourse markers as a feature of online language production.

### 17. hedge_rate
- **What:** Frequency of hedging expressions per sentence
- **Formula:** `count(hedges) / n_sentences`
- **Hedges:** {i think, i guess, maybe, perhaps, probably, kind of, sort of, i believe, it seems, i suppose, might be}
- **Why:** Genuine speakers hedge uncertain statements ("I think maybe..."). Pre-written answers are assertive and definitive.
- **Sources:** Lakoff (1973) — "Hedges: A study in meaning criteria", *Journal of Philosophical Logic*. Holmes (1990) — hedging in spoken discourse.

### 18. noun_rate
- **What:** Proportion of noun tokens
- **Formula:** `count(NOUN POS) / total_tokens` (spaCy POS tagging)
- **Why:** Content-word density varies between planned and unplanned speech. Scripted text is noun-heavy (informational); spontaneous speech uses more function words.
- **Sources:** Biber (1988) — noun frequency as a dimension of register variation between speech and writing.

### 19. verb_rate
- **What:** Proportion of verb tokens
- **Formula:** `count(VERB POS) / total_tokens`
- **Why:** Spoken language tends to be more verb-heavy (action/narrative) while written text is more noun-heavy (descriptive).

### 20. adj_rate
- **What:** Proportion of adjective tokens
- **Formula:** `count(ADJ POS) / total_tokens`
- **Why:** Descriptive, prepared answers use more adjectives. Genuine speech is more action-oriented.

---

## Pause Features (13)

Extracted from word-level timestamps (WhisperX/faster-whisper alignment). A pause is any gap >= 0.05 seconds between consecutive words.

### 21. pause_mean
- **What:** Average pause duration (seconds)
- **Why:** Longer average pauses may indicate reading ahead or cognitive load from deception. Genuine speakers have shorter, more natural pauses.
- **Sources:** Goldman-Eisler (1968) — "Psycholinguistics: Experiments in Spontaneous Speech". Foundational work on pause distribution in speech production.

### 22. pause_std
- **What:** Standard deviation of pause durations
- **Why:** Consistent pauses (low std) suggest mechanical/scripted delivery. Variable pauses (high std) suggest natural thought processes.

### 23. pause_median
- **What:** Median pause duration
- **Why:** Robust to outlier pauses (e.g., one very long pause from external interruption). More representative of typical pause behavior.

### 24. pause_skew
- **What:** Skewness of pause duration distribution
- **Formula:** `pandas.Series(pauses).skew()`
- **Why:** Positive skew (many short pauses, few long) is typical of natural speech. Scripts may show different distributional shape.
- **Sources:** Campione & Veronis (2002) — "A large-scale multilingual study of silent pause duration", *Speech Prosody*. Characterized pause distribution shapes across languages.

### 25. long_pause_rate
- **What:** Proportion of pauses longer than 0.5 seconds
- **Formula:** `count(pauses > 0.5s) / n_pauses`
- **Why:** Long pauses (>0.5s) suggest active thinking or reading ahead. Genuine speakers fill long gaps with fillers; readers pause silently while scanning text.
- **Sources:** Brennan & Williams (1995) — "The feeling of another's knowing", *Journal of Memory and Language*. Connected pause duration to cognitive processing.

### 26. pause_ratio
- **What:** Proportion of total time spent pausing
- **Formula:** `sum(pause_durations) / total_duration`
- **Why:** Higher pause ratio = more time silent. Can indicate looking at a script or processing written text before speaking.

### 27. n_pauses
- **What:** Total count of pauses (gaps >= 0.05s)
- **Why:** Baseline count. Genuine speech typically has more frequent but shorter pauses.

### 28. pause_regularity
- **What:** Standard deviation of intervals between consecutive pauses
- **Formula:** `std(positions_of_pauses)` — measures how evenly spaced pauses are
- **Why:** Natural speech has irregular pause placement (governed by thought). Scripted reading produces more regularly-spaced pauses (governed by punctuation/line breaks).
- **Sources:** Grosjean & Deschamps (1975) — "Analyse contrastive des variables temporelles de l'anglais et du francais", *Phonetica*. Analyzed temporal regularity in speech.

### 29. pause_before_content_ratio
- **What:** Proportion of pauses occurring before content words (NOUN, VERB, ADJ, ADV, PROPN)
- **Formula:** `count(pauses before content words) / n_pauses`
- **Why:** Readers pause before content words because they're scanning/processing the next meaningful word. Genuine speakers pause at phrase boundaries regardless of word type.
- **Sources:** Beattie & Butterworth (1979) — "Contextual probability and word frequency as determinants of pauses in spontaneous speech", *Language and Speech*. Found pauses cluster before low-frequency content words during lexical retrieval.

### 30. pause_before_function_ratio
- **What:** Proportion of pauses before function words (DET, ADP, CONJ, PRON, AUX, PART)
- **Formula:** `count(pauses before function words) / n_pauses`
- **Why:** Complement to content ratio. Natural speech pauses occur at syntactic boundaries, often before function words that start new phrases.

### 31. mid_phrase_pause_rate
- **What:** Proportion of pauses occurring mid-phrase (not after sentence-ending punctuation)
- **Formula:** `count(pauses where previous word doesn't end with .,!?,) / n_pauses`
- **Why:** Genuine speakers pause at clause/sentence boundaries. Readers pause mid-phrase when scanning ahead, creating unnatural break points.
- **Sources:** Zellner (1994) — "Pauses and the temporal structure of speech", in *Fundamentals of Speech Synthesis and Speech Recognition*. Differentiated boundary vs. hesitation pauses.

### 32. words_per_sec
- **What:** Overall speaking rate including pauses
- **Formula:** `n_words / total_duration`
- **Why:** Scripted reading often has an unnaturally consistent or fast rate. Spontaneous speech varies more.
- **Sources:** Goldman-Eisler (1968). Levelt (1989) — "Speaking: From Intention to Articulation", MIT Press. Both documented speaking rate as a production variable.

### 33. articulation_rate
- **What:** Speaking rate excluding pauses (pure speech segments only)
- **Formula:** `n_words / (total_duration - sum(pause_durations))`
- **Why:** Separates pause behavior from articulation speed. Some cheaters read quickly between pauses, producing high articulation rate but normal words_per_sec.
- **Sources:** Grosjean & Deschamps (1975). Tsao & Weismer (1997) — "Interspeaker variation in habitual speaking rate", *JSLHR*.

---

## Prosodic Features (8)

Extracted from raw audio using librosa. F0 (fundamental frequency/pitch) via `librosa.pyin` (75-500 Hz range, frame_length=2048). Energy via `librosa.feature.rms`.

### 34. f0_mean
- **What:** Mean fundamental frequency across voiced frames (Hz)
- **Why:** Baseline pitch. Not discriminative on its own but necessary for context. Combined with other F0 features captures pitch behavior.
- **Sources:** Banse & Scherer (1996) — "Acoustic profiles in vocal emotion expression", *JASA*. F0 as a core vocal parameter.

### 35. f0_std
- **What:** Standard deviation of F0 across voiced frames
- **Why:** Scripted reading often has reduced pitch variation (monotone). Genuine spontaneous speech shows wider pitch modulation for emphasis and emotion.
- **Sources:** Hirschberg et al. (2005) — "Distinguishing deceptive from non-deceptive speech", *Interspeech*. Found F0 variability differs between deceptive and truthful speech.

### 36. f0_range
- **What:** Pitch range (max F0 - min F0 in Hz)
- **Why:** Similar to f0_std but captures extremes. Expressive/genuine speech has wider range. Monotone reading has narrow range.

### 37. f0_skew
- **What:** Skewness of F0 distribution
- **Formula:** `pandas.Series(f0_voiced).skew()`
- **Why:** Captures asymmetry in pitch distribution. Natural speech often has positive skew (mostly lower pitch with occasional high peaks for emphasis).

### 38. f0_slope
- **What:** Linear trend of pitch across the utterance
- **Formula:** `np.polyfit(x, f0_voiced, 1)[0]` — first coefficient (slope)
- **Why:** Natural speech typically shows downward pitch declination within utterances (related to breath). Scripted reading may show different pitch contours.
- **Sources:** Lieberman (1967) — "Intonation, Perception, and Language". 't Hart et al. (1990) — "A Perceptual Study of Intonation". Both documented declination as a universal speech production phenomenon.

### 39. energy_mean
- **What:** Mean RMS energy across all audio frames
- **Formula:** `librosa.feature.rms(frame_length=512, hop_length=256).mean()`
- **Why:** Overall loudness. Confident genuine speakers may have higher energy. Whispering while cheating or reading in a test environment may lower energy.

### 40. energy_std
- **What:** Standard deviation of RMS energy
- **Why:** Energy variation captures dynamic range. Natural speech modulates loudness for emphasis. Monotone reading has more uniform energy.
- **Sources:** Atal (1972) — early work on energy contours in speech. Mo et al. (2009) — "The effects of word predictability on the time course of speaking", *JASA*.

### 41. speaking_rate_std
- **What:** Variability in local speaking rate across 2-second windows
- **Formula:** For each 2-second chunk, compute proportion of frames above 20th-percentile RMS (proxy for speech activity); take SD across all chunks
- **Why:** Natural speakers speed up and slow down. Readers maintain more constant pace. This captures the temporal dynamics that a global words_per_sec misses.
- **Sources:** Quene (2008) — "Multilevel modeling of between-speaker and within-speaker variation in spontaneous speech tempo", *JASA*.

---

## General Academic Grounding

The overall approach of using linguistic, temporal, and prosodic features to distinguish spontaneous from read/scripted speech draws from several research traditions:

1. **Deception detection:** DePaulo et al. (2003) — "Cues to deception", *Psychological Bulletin*. Meta-analysis of verbal and nonverbal cues to deception.

2. **Speaking style classification:** Laan (1997) — "The contribution of intonation, segmental durations, and spectral features to the perception of a spontaneous and a read speaking style". Showed acoustic differences between read and spontaneous speech.

3. **Disfluency research:** Shriberg (2001) — "To 'errrr' is human: Ecology and acoustics of speech disfluencies", *Journal of the International Phonetic Association*. Comprehensive framework for speech disfluencies.

4. **Computational paralinguistics:** Schuller et al. (2013) — "Computational Paralinguistics: Emotion, Affect and Personality in Speech and Language Processing". Framework for extracting paralinguistic information from speech.

5. **Read vs. spontaneous speech:** Howell & Kadi-Hanifi (1991) — "Comparison of prosodic properties between read and spontaneous speech material", *Speech Communication*. Direct comparison establishing measurable differences.
