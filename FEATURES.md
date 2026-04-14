# Feature Reference: Cheating Detection v3

All features used in the text+audio XGBoost model. Total: **49 features** (21 text + 17 pause + 8 prosodic + 3 voice quality + 2 perplexity) + **768 WavLM dims**.

---

## 1. Text Features (21)

Extracted from Whisper transcript. Capture **how** the candidate speaks, not what they say.

| # | Feature | Full Form | How Calculated | Reasoning | Expected Direction |
|---|---|---|---|---|---|
| 1 | `filler_rate` | Filler Word Rate | Count of {um, uh, uhm, hmm, hm, er, ah, mhm} / total word count | Fillers are thinking tokens — genuine speakers produce them while formulating thoughts. Readers have none because the text is already formulated | Genuine > Cheating |
| 2 | `filler_count` | Filler Word Count | Raw count of filler words in transcript | Absolute count complements rate — short answers with even 2-3 fillers are significant | Genuine > Cheating |
| 3 | `ttr` | Type-Token Ratio | Unique words / Total words | Vocabulary diversity. Readers use planned, non-repetitive vocabulary. Genuine speakers repeat words while thinking | Cheating > Genuine |
| 4 | `mattr` | Moving Average Type-Token Ratio | Mean of TTR over sliding 50-word windows | Length-independent version of TTR — avoids the bias where longer texts mechanically have lower TTR | Cheating > Genuine |
| 5 | `mtld` | Measure of Textual Lexical Diversity | Average run length before TTR drops below 0.72 threshold, computed forward and backward and averaged | Most length-independent lexical diversity metric. Readers have high MTLD (planned, varied vocab). Spontaneous speakers repeat words → lower MTLD | Cheating > Genuine |
| 6 | `complex_word_rate` | Complex Word Rate | Words with ≥3 syllables / total words | Written/AI text uses complex vocabulary. Spontaneous speech skews simpler | Cheating > Genuine |
| 7 | `avg_word_length` | Average Word Length | Mean character count per word | Correlated with vocabulary sophistication. Prepared text uses longer words | Cheating > Genuine |
| 8 | `n_words` | Word Count | Total alpha tokens from spaCy tokenization | Controls for response length — very short or very long responses have different patterns | Context feature |
| 9 | `n_unique_words` | Unique Word Count | Count of distinct word types | Raw vocabulary size; high for prepared answers with wide vocabulary | Cheating > Genuine |
| 10 | `avg_sentence_length` | Average Sentence Length | Mean word count per spaCy-detected sentence | AI/prepared text tends toward longer, complete sentences. Genuine speech: fragments, shorter bursts | Cheating > Genuine |
| 11 | `std_sentence_length` | Standard Deviation of Sentence Length | Std dev of per-sentence word counts | Genuine speakers vary sentence length wildly (fragments + run-ons). Prepared text more uniform | Genuine > Cheating |
| 12 | `fragment_rate` | Fragment Rate | Sentences with <4 words / total sentences | Incomplete sentences are a spontaneous speech marker — trail-offs, restarts, short affirmations | Genuine > Cheating |
| 13 | `n_sentences` | Sentence Count | spaCy sentence boundary count | Structure proxy; prepared text is organized into coherent sentences | Cheating > Genuine |
| 14 | `repetition_rate` | Repetition Rate | (Bigram count - 1) summed over repeated bigrams / total bigrams | Genuine speakers repeat phrases while thinking ("I think, I think"). Readers don't | Genuine > Cheating |
| 15 | `repair_rate` | Repair Rate | Count of repair phrases {I mean, no wait, sorry I, actually no} / sentences | Self-corrections are spontaneous speech markers — genuine speakers change course mid-sentence | Genuine > Cheating |
| 16 | `self_ref_rate` | Self-Reference Rate | Tokens in {I, me, my, myself, I'm, I've} / total words | Genuine answers are personal and self-referential. Prepared/AI text is more impersonal/declarative | Genuine > Cheating |
| 17 | `discourse_marker_rate` | Discourse Marker Rate | Count of {you know, I mean, like, basically, so, well, right, okay} / sentences | Discourse markers are speech-planning devices used in real-time, not in written text | Genuine > Cheating |
| 18 | `hedge_rate` | Hedge Rate | Count of {I think, I guess, maybe, perhaps, probably, kind of, sort of} / sentences | Epistemic hedging signals genuine uncertainty — people hedging while speaking spontaneously. Prepared answers assert confidently | Genuine > Cheating |
| 19 | `noun_rate` | Noun Rate | NOUN/PROPN token count / total POS tokens (spaCy) | Written text is noun-heavy (nominalization). Spoken language is verb-heavy | Cheating > Genuine |
| 20 | `verb_rate` | Verb Rate | VERB token count / total POS tokens | Spontaneous speech is verb-heavy — action-driven, real-time narration | Genuine > Cheating |
| 21 | `adj_rate` | Adjective Rate | ADJ token count / total POS tokens | Descriptive adjectives more common in planned prose | Cheating > Genuine |

**Sources:**
- Biber, D. (1988). *Variation across Speech and Writing*. Cambridge University Press. (Systematic feature differences between spoken and written register)
- Chafe, W. (1982). Integration and involvement in speaking, writing, and oral literature. *Spoken and Written Language*, 35-53. (Self-reference, hedging, fragmentation as spoken markers)
- McCarthy, P.M. & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381-392. (MTLD methodology)

---

## 2. Pause Features (17)

Extracted from **Whisper word-level timestamps** (not from transcription text). Capture **when** the candidate pauses and what the pauses mean.

| # | Feature | Full Form | How Calculated | Reasoning | Expected Direction |
|---|---|---|---|---|---|
| 1 | `pause_mean` | Mean Pause Duration | Mean of all inter-word gaps > 50ms | Readers have short, uniform pauses at punctuation. Genuine speakers have longer, more variable pauses | Genuine > Cheating |
| 2 | `pause_std` | Pause Standard Deviation | Std dev of inter-word gaps > 50ms | Variability of pauses. Reading = metronomic, genuine = irregular | Genuine > Cheating |
| 3 | `pause_median` | Median Pause Duration | Median of inter-word gaps > 50ms | Robust to outlier pauses; central tendency of pause behavior | Genuine > Cheating |
| 4 | `pause_skew` | Pause Duration Skew | Skewness of pause distribution | Genuine speech has positively skewed pauses (mostly short, occasional very long thinking pauses). Reading is more symmetric | Genuine > Cheating |
| 5 | `long_pause_rate` | Long Pause Rate | Pauses > 500ms / total pauses | Thinking pauses are typically >500ms. Readers don't need to think | Genuine > Cheating |
| 6 | `pause_ratio` | Pause-to-Speech Ratio | Total pause duration / total response duration | What fraction of the response is silence. High in genuine speakers (thinking time) | Genuine > Cheating |
| 7 | `n_pauses` | Number of Pauses | Count of inter-word gaps > 50ms | Raw count of hesitation events | Genuine > Cheating |
| 8 | `pause_regularity` | Pause Regularity | Std dev of inter-pause intervals (positions between pauses) | Reading has regularly-spaced pauses (sentence boundaries). Genuine speech: pauses appear at irregular positions | Cheating > Genuine |
| 9 | `pause_before_content_ratio` | Pause Before Content Word Ratio | Pauses preceding NOUN/VERB/ADJ/ADV/PROPN / total pauses | Genuine speakers pause *before* content words (searching for the right word). Readers pause *after* content (at punctuation) | Genuine > Cheating |
| 10 | `pause_before_function_ratio` | Pause Before Function Word Ratio | Pauses preceding DET/ADP/CONJ/PRON / total pauses | Pausing before function words is unusual — suggests reading pace | Cheating > Genuine |
| 11 | `mid_phrase_pause_rate` | Mid-Phrase Pause Rate | Pauses not following sentence-ending punctuation / total pauses | Genuine speakers pause mid-thought. Readers pause at sentence ends | Genuine > Cheating |
| 12 | `words_per_sec` | Words Per Second | Total words / total response duration | Speech rate. Reading tends to be faster (text already prepared) | Cheating > Genuine |
| 13 | `articulation_rate` | Articulation Rate | Total words / (total duration - total pause duration) | Speech rate excluding silence — pure speaking speed without pauses | Cheating > Genuine |
| 14 | `initial_pause` | Initial Pause | Timestamp of first word start | Time between question end and first word. Genuine speakers pause to think; readers find their place quickly | Genuine > Cheating |
| 15 | `longest_pause` | Longest Pause Duration | Maximum inter-word gap > 50ms in the response | Longest thinking pause. Genuine speakers occasionally have very long pauses (searching for ideas) | Genuine > Cheating |
| 16 | `suspicious_gap_count` | Suspicious Gap Count | Count of 0.3-0.8s gaps that are NOT at sentence boundaries | Mid-sentence gaps of this duration are where fillers would normally appear. Their absence with a gap suggests the filler was suppressed/edited | Cheating > Genuine |
| 17 | `suspicious_gap_ratio` | Suspicious Gap Ratio | Suspicious gap count / total word count | Normalized version of suspicious gap count | Cheating > Genuine |

**Sources:**
- Shriberg, E. (2001). To 'errrr' is human: Ecology and acoustics of speech disfluencies. *Journal of the International Phonetic Association*, 31(1), 153-169.
- Kosmala, L. & Meunier, C. (2017). Pauses in native and non-native spontaneous speech. *Interspeech 2017*.
- Rieger, C.L. (2003). Repetitions as self-repair strategies in English and German conversations. *Journal of Pragmatics*, 35(1), 47-69.

---

## 3. Prosodic Features (8)

Extracted from **raw audio** using librosa. Capture **pitch and energy patterns** across the response.

| # | Feature | Full Form | How Calculated | Reasoning | Expected Direction |
|---|---|---|---|---|---|
| 1 | `f0_mean` | Mean Fundamental Frequency | Mean of voiced F0 frames (librosa pyin, 75-500 Hz range) | Average pitch. Reading tends toward a narrower pitch range; genuine speech more expressive | Context feature |
| 2 | `f0_std` | F0 Standard Deviation | Std dev of voiced F0 frames | Pitch variability. Genuine = higher variation (emphasis, questions, thinking). Reading = more monotone | Genuine > Cheating |
| 3 | `f0_range` | F0 Range | max(F0) - min(F0) across voiced frames | Dynamic range of pitch. Wide range = expressive, spontaneous speech | Genuine > Cheating |
| 4 | `f0_skew` | F0 Skewness | Skewness of the F0 distribution | Shape of pitch distribution. Reading tends toward more symmetric/normal F0 distribution | Context feature |
| 5 | `f0_slope` | F0 Slope | Linear regression slope over time of F0 values | Overall pitch trend across response. Genuine speech may have more dramatic declination | Context feature |
| 6 | `energy_mean` | Mean RMS Energy | Mean root-mean-square energy (frame_length=512) | Average loudness across the response | Context feature |
| 7 | `energy_std` | Energy Standard Deviation | Std dev of RMS energy frames | Energy variability — genuine speech is more dynamically varied | Genuine > Cheating |
| 8 | `speaking_rate_std` | Speaking Rate Variability | Std dev of per-2s-window voiced frame ratios | How much speaking rate fluctuates across the response. Genuine speakers speed up/slow down; readers maintain pace | Genuine > Cheating |

**Sources:**
- Laan, G.P.M. (1997). The contribution of intonation, segmental durations, and spectral features to the perception of a spontaneous and read speech style. *Journal of the Acoustical Society of America*, 101(4), 2key-2405.
- Batliner, A. et al. (2011). *The PF_STAR Children's Speech Corpus*. (Prosodic features for speech style classification)

---

## 4. Voice Quality Features (3)

Extracted from **raw audio** using Parselmouth (Python wrapper for Praat). Capture **vocal fold behaviour** — a physiological signal of cognitive load and speaking style.

| # | Feature | Full Form | How Calculated | Reasoning | Expected Direction |
|---|---|---|---|---|---|
| 1 | `jitter_local` | Local Jitter | Cycle-to-cycle variation in F0 period: mean(|T_i - T_{i-1}|) / mean(T_i) using Praat PointProcess | Vocal fold control. Reading = controlled breathing and articulation → low jitter. Genuine = higher cognitive load → less precise vocal fold control → higher jitter | Genuine > Cheating |
| 2 | `shimmer_local` | Local Shimmer | Cycle-to-cycle variation in amplitude: mean(|A_i - A_{i-1}|) / mean(A_i) | Same principle as jitter but for amplitude. Prepared reading = steady amplitude. Spontaneous = variable | Genuine > Cheating |
| 3 | `hnr_mean` | Harmonics-to-Noise Ratio Mean | 10 * log10(energy of harmonic component / energy of noise component), averaged via Praat Harmonicity object | Measure of voice "cleanliness". Reading = clear, controlled phonation → high HNR. Genuine = breathier voice (cognitive load, faster breathing) → lower HNR | Cheating > Genuine |

**Library:** `praat-parselmouth` — Python bindings for Praat, the gold-standard speech analysis software used in phonetics research since 1992.

**Sources:**
- Farrús, M., Hernando, J. & Ejarque, P. (2007). Jitter and shimmer measurements for speaker recognition. *Interspeech 2007*, 778-781.
- Baken, R.J. & Orlikoff, R.F. (2000). *Clinical Measurement of Speech and Voice* (2nd ed.). Singular. (Clinical reference for jitter/shimmer/HNR interpretation)
- Eyben, F. et al. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPSv01b). *IEEE Transactions on Affective Computing*, 7(2), 190-202. (jitter, shimmer, HNR all included in eGeMAPS standard)

---

## 5. Perplexity Features (2)

Extracted from **Whisper transcript** using GPT-2 (124M parameters, runs on CPU). Capture **how predictable the text is** — direct signal for AI-generated content.

| # | Feature | Full Form | How Calculated | Reasoning | Expected Direction |
|---|---|---|---|---|---|
| 1 | `mean_perplexity` | Mean GPT-2 Perplexity | For each sentence: compute cross-entropy loss under GPT-2, exponentiate to get per-sentence perplexity. Average across sentences. PPL = exp(-1/N * Σ log P(w_i)) | GPT-2 assigns high probability to text that looks like AI output (low perplexity). Human spontaneous speech is disfluent and unpredictable (high perplexity). AI-generated answers score PPL ~10-40; genuine speech transcripts ~60-200 | Genuine > Cheating |
| 2 | `burstiness` | Perplexity Burstiness | Variance of per-sentence perplexity scores across the response | AI-generated text has uniform complexity sentence-to-sentence (low variance). Human speech alternates between simple and complex utterances wildly (high variance). Named "burstiness" after the information theory concept | Genuine > Cheating |

**Sources:**
- Lavergne, T. et al. (2008). Practical very large scale CRFs. *ACL 2010*. (Perplexity as a detection signal)
- Gehrmann, S., Strobelt, H. & Rush, A.M. (2019). GLTR: Statistical detection and visualization of generated text. *ACL 2019*. (GPT-2 perplexity for AI text detection)
- Ippolito, D. et al. (2020). Automatic detection of generated text is easiest when humans are fooled. *ACL 2020*. (Perplexity + burstiness combination)
- GPTZero methodology (Edward Tian, 2023) — commercial AI detector uses same two features (perplexity + burstiness) as primary signals

---

## 6. WavLM Embeddings (768 dims)

Not hand-engineered features — a pretrained deep speech representation.

| Signal | Full Form | How Calculated | Reasoning |
|---|---|---|---|
| `wavlm_0` ... `wavlm_767` | WavLM Base-Plus Hidden States | Load audio at 16kHz → WavLM feature extractor → mean-pool last hidden state across time frames → 768-dim vector | WavLM (Microsoft, 2022) is pretrained on 94,000 hours of speech via masked speech prediction. The resulting embeddings capture acoustic patterns at a level of abstraction unreachable by hand-engineered features. Acts as a catch-all for signals we can't explicitly name |

**Why no PCA:** Tried PCA 768→80 dims, performance dropped from ~85% to ~58% F1. The PCA projection destroys discriminative signal that is spread across many dimensions. Instead, `colsample_bytree=0.2` in XGBoost randomly samples ~154 dims per tree — same regularization effect without information loss.

**Model:** `microsoft/wavlm-base-plus` (HuggingFace). 94M parameters.

**Sources:**
- Chen, S. et al. (2022). WavLM: Large-scale self-supervised pre-training for full stack speech processing. *IEEE Journal of Selected Topics in Signal Processing*, 16(6), 1505-1518.
- Pappagari, R. et al. (2020). Copycatch: Detecting audio deepfakes with WavLM. (Application of WavLM embeddings to audio classification)

---

## 7. Why These Specific Features Together

Each feature group targets a **different cheating strategy**:

| Cheating Type | Primary Signal | Secondary Signal |
|---|---|---|
| Reading GPT output | Low perplexity + burstiness | No fillers, high MTLD, low jitter |
| Reading article/notes | No fillers, fast speech, low pause_std | High noun_rate, high MTLD, metronomic pauses |
| Pre-written bullet notes | Suspicious gaps (checkpoint pauses) | High initial_pause, low repair_rate |
| Live help (someone feeding answers) | Long initial_pause, then fluent | Low filler_rate, unusual WavLM pattern |
| **Genuine answer** | High filler_rate, mid-phrase pauses | High perplexity, low MTLD, high jitter |

No single feature catches all types. The multi-signal fusion is the core design principle.

---

## 8. Feature Extraction Libraries

| Library | Version | Purpose |
|---|---|---|
| `faster-whisper` | ≥0.10 | Transcription + word timestamps |
| `spacy` (en_core_web_sm) | ≥3.0 | Tokenization, POS tagging, sentence segmentation |
| `librosa` | ≥0.10 | Prosodic feature extraction (F0, RMS energy) |
| `praat-parselmouth` | ≥0.4.3 | Voice quality (jitter, shimmer, HNR) |
| `transformers` (GPT-2) | ≥4.0 | Perplexity and burstiness |
| `torch` | ≥2.0 | GPT-2 and WavLM inference |
| `transformers` (WavLM) | ≥4.0 | 768-dim audio embeddings |
