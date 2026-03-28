# Research: Detecting Prepared/Pre-Written Speech vs Authentic Spontaneous Speech
# Comprehensive Multi-Signal Approach for Exam/Interview Cheating Detection
# Research compiled: March 2026

---

## Executive Summary

The task of detecting when a candidate is reading from prepared material (GPT output, articles, pre-written notes) vs genuinely thinking and answering requires a **multi-signal approach** combining:

1. **Linguistic/text features** (strongest signal for your specific problem)
2. **Pause and timing analysis** (where pauses occur, not just how many)
3. **Disfluency patterns** (filled pauses, self-corrections, false starts)
4. **Prosodic features** (pitch variability, speaking rate variance)
5. **AI-generated text detection** (perplexity-based analysis of transcripts)

No single signal is sufficient. The recommended architecture is a **late-fusion ensemble** combining CrisperWhisper verbatim transcription with acoustic feature extraction, feeding into XGBoost/LightGBM.

---

## 1. Academic Research: Read vs Spontaneous Speech Classification

### 1.1 The Definitive Paper (December 2024, SLT Workshop)

**"Classification of Spontaneous and Scripted Speech for Multilingual Audio"**
(arXiv:2412.11896, Elisha et al., Spotify Research + Queen Mary University)

**What they tested:**
| Model | Scripted F1 | Spontaneous F1 | AUC |
|-------|------------|----------------|-----|
| eGeMAPSv02 (88 acoustic features) | 0.69 | 0.86 | 0.87 |
| Handcrafted (115 features incl. speaking rate, overlap) | 0.76 | 0.88 | 0.91 |
| YAMNet (pre-trained audio classifier) | 0.70 | 0.87 | 0.90 |
| Whisper encoder (frozen, last hidden state) | **0.83** | **0.91** | **0.95** |
| Fine-tuned Whisper | 0.82 | 0.91 | 0.95 |

**Cross-domain generalization (AUC):**
| Model | Podcasts | CEFC (French) | DIHARD (English/Mandarin) |
|-------|---------|---------------|---------------------------|
| eGeMAPSv02 | 0.87 | 0.53 | 0.46 |
| Handcrafted | 0.91 | 0.74 | 0.59 |
| Whisper encoder | **0.95** | **0.92** | **0.95** |

**Key findings:**
- Whisper encoder (frozen) vastly outperforms handcrafted features, especially cross-domain
- Fine-tuning Whisper provided NO improvement (representations already good enough)
- Handcrafted features collapse when domain changes (0.91 -> 0.46-0.59)
- Most important handcrafted features: speech/silence duration distributions, speaking rate stats, overlapping speech detection
- Embedding size trade-off: handcrafted = 4KB, Whisper = 7.3MB per sample
- Japanese was anomalous (mora-timed language), suggesting rhythm-class sensitivity

**Limitations for our task:**
- Their task (podcast scripted ads vs conversation) has cleaner acoustic distinction than our task (candidate reading AI answer vs genuine answering)
- They don't analyze linguistic content at all
- Dataset is professionally recorded podcasts, not noisy assessment recordings

**Production viability:** Whisper-large-v2 encoder is heavy (~1.5GB). Whisper-small encoder (~200MB ONNX) should work for our use case with some accuracy loss.

### 1.2 Prosody-Only Classification (Batliner et al.)

**"Can You Tell Apart Spontaneous and Read Speech if You Just Look at Prosody?"**

- Speaker-independent classification using only prosodic features: **55-65% accuracy**
- The distinction is "rather complex and partly speaker dependent"
- F0 patterns differ but overlap substantially between speaking styles
- **This is essentially what pure Wav2Vec2/WavLM models do** -- and matches your observed 45-60% on company data

**Implication:** Prosody alone is insufficient for our task. Must combine with linguistic features.

### 1.3 F0 and Pitch Patterns

**Key research findings on pitch differences:**
- Read speech has **steeper and more frequent F0 declination** than spontaneous speech
- Spontaneous speech has **higher mean F0, higher SD, higher skewness** (speakers are more "lively")
- Spontaneous speech has **larger F0 excursions** and greater overall variability
- F0 slopes are shallower in spontaneous than read speech
- One study: 93.7% accuracy on read segments, only 69.5% on spontaneous (asymmetric difficulty)

**Practical feature extraction:**
- F0 mean, std, range, skewness (per utterance)
- F0 declination slope
- F0 reset frequency (how often pitch resets at boundaries)
- Can extract with PRAAT, librosa, or openSMILE

### 1.4 Residual Speech Embeddings (Feb 2025)

**arXiv:2502.19387 - "Removing Linguistic Content to Enhance Paralinguistic Analysis"**

- Novel technique: regress speech embeddings onto text embeddings, use the **residual** (what's left after removing linguistic content)
- WavLM residual embeddings: 98-100% accuracy on tone classification
- Separates "what was said" from "how it was said"
- **Highly relevant for our task** where we need BOTH signals independently

**Limitation:** Adds pipeline complexity (need ASR + text embedding + speech embedding + residual computation)

---

## 2. Pause Analysis: The Most Promising Acoustic Signal

### 2.1 Pause Structure Differences (Cernak et al., EURASIP 2016)

**"Structure of Pauses in Speech in Context of Speaker Verification and Classification of Speech Type"**

**Key findings:**
- In **read speech**: ALL pauses are "planned" -- structurally determined, encoding prosodic structure (at clause boundaries, commas, sentence ends)
- In **spontaneous speech**: Two types of pauses exist:
  - **Grammatical pauses** (at boundaries, similar to read speech)
  - **Non-grammatical/unplanned pauses** (for word-finding, speech planning -- occur mid-phrase, before content words)
- Spontaneous speech pauses are **much longer** and more frequent than read speech pauses
- Pause duration distributions differ significantly between the two styles

**Critical distinguishing feature:**
> In spontaneous speech, pauses occur more before **content words** (nouns, verbs) than function words, and are longer before verbs relative to nouns. This reflects real-time lexical access and speech planning.

> In read speech, pauses occur at **syntactic boundaries** (sentence ends, commas), NOT before content words.

**This is perhaps the single most powerful acoustic feature for our task:**
- Someone genuinely thinking will pause before the difficult content word they're trying to recall
- Someone reading will pause only at punctuation/breathing points
- This can be measured with word-level timestamps from CrisperWhisper

### 2.2 Measuring Pause Patterns

**Practical features to extract:**
1. **Pause-before-content-word ratio**: % of pauses that occur before nouns/verbs/adjectives
2. **Pause-before-function-word ratio**: % of pauses that occur before determiners/prepositions/conjunctions
3. **Mid-phrase pause rate**: pauses NOT at clause/sentence boundaries
4. **Pause duration distribution**: mean, std, skewness of pause lengths
5. **Pause regularity**: std of inter-pause intervals (reading = more regular, spontaneous = irregular)
6. **Long pause rate**: % of pauses > 500ms (more in spontaneous)
7. **Speaking rate variability**: std of syllable rate across chunks

**How to compute:**
```
CrisperWhisper -> word-level timestamps + POS tags (via spaCy)
-> For each pause > 150ms:
   - What word follows? Content or function?
   - Is it at a clause boundary?
   - Duration?
-> Aggregate into features
```

### 2.3 Cognitive Load and Pause Placement

**Research on pause-cognition relationship:**
- Pauses before nouns are specifically driven by **lemma access difficulty** (finding the right word)
- Filled pauses ("um", "uh") occur most when **difficult material is still being planned**
- Repeats occur when **difficult material is already being produced** and can be repeated
- Speakers under higher cognitive load produce **more and longer pauses**, especially at clause boundaries
- Semantic working memory (not phonological) drives content word planning pauses

**Implication for cheating detection:**
- A genuinely thinking candidate has HIGH cognitive load -> more pauses before content words, more filled pauses during planning
- A reading candidate has LOW cognitive load -> pauses only at punctuation, no filled pauses

---

## 3. Disfluency Analysis: The Spontaneity Fingerprint

### 3.1 Types and Frequencies

**Disfluencies in spontaneous speech (typical rates):**
| Type | Rate in spontaneous speech | Rate in read speech |
|------|---------------------------|---------------------|
| Filled pauses ("um", "uh") | 1.8% of all words, 22.9% of disfluencies | Near zero |
| Repetitions ("I I I think") | Common | Near zero |
| False starts / repairs ("I went-- I drove") | ~10% of utterances | Near zero |
| Lengthenings ("sooo", "aaand") | Common | Rare |
| Self-corrections | Common | Near zero |

**Key insight:** Self-repairs occur in about **10% of spontaneous utterances**. Their near-total absence is a strong indicator of reading.

### 3.2 Detecting Disfluencies Automatically

**CrisperWhisper (Interspeech 2024, Zusag et al.):**
- Specifically designed for **verbatim transcription** including all disfluencies
- Transcribes fillers ("um", "uh"), stutters, false starts, partial words
- F1-score of **84.7%** for word-level timestamps (vs WhisperX's 76.7%)
- Near-perfect filler detection on PodcastFillers corpus
- Robust under noise: 79.5% F1 vs WhisperX's 59.0% with background noise
- Reduces WER from 16.82% to 9.72% on AMI Meeting Corpus
- **Open source**: github.com/nyrahealth/CrisperWhisper

**Why CrisperWhisper over standard Whisper:**
- Standard Whisper is trained to OMIT disfluencies (produces "clean" transcripts)
- Standard Whisper's timestamps are inaccurate by several seconds at utterance level
- CrisperWhisper preserves every spoken element -- critical for our disfluency analysis
- Word-level timestamp accuracy enables precise pause measurement

**Comparison of Whisper variants for our task:**

| Feature | Whisper | WhisperX | whisper-timestamped | CrisperWhisper |
|---------|---------|----------|--------------------|-----------------|
| Verbatim transcription | No (omits fillers) | No | Partial | **Yes** |
| Word timestamps | Utterance-level only | Via forced alignment | Via DTW | **Via improved DTW** |
| Timestamp accuracy | Poor | Good | Good | **Best** |
| Filler detection | No | No | Partial | **Yes** |
| Noise robustness | Good | Poor | Medium | **Good** |
| Disfluency types | None | None | Some | **All** |
| Extra model needed | No | Yes (wav2vec2) | No | No |

**CrisperWhisper is the clear winner for our pipeline.**

### 3.3 Features to Extract from Disfluencies

1. **Filled pause rate**: count("um", "uh", "hmm", "er") / total_words
2. **Repetition rate**: repeated bigrams or trigrams / total n-grams
3. **False start rate**: count of abandoned utterances / total utterances
4. **Self-correction rate**: count of repairs ("I mean", restarts) / total utterances
5. **Lengthening rate**: elongated words / total words
6. **Discourse marker rate**: "you know", "like", "basically", "so", "well" / total words
7. **Filler-to-pause ratio**: filled pauses / total pauses (higher = more spontaneous)

---

## 4. Linguistic/Text Features: The Strongest Signal

### 4.1 Why Text Features Matter Most for This Task

Your own observation confirmed this: cheaters use complex vocabulary and sentence structure, while genuine answerers use common words and repeat themselves. This is a **content-level signal** invisible to acoustic-only models.

### 4.2 Vocabulary Complexity Features

| Feature | What it measures | Expected: Cheating | Expected: Genuine |
|---------|-----------------|--------------------|--------------------|
| Type-Token Ratio (TTR) | Vocabulary diversity | Higher (varied vocab) | Lower (repeated words) |
| MATTR (Moving Average TTR) | Length-normalized diversity | Higher | Lower |
| Avg word length | Vocabulary sophistication | Longer | Shorter |
| Complex word rate | Words with 3+ syllables | Higher | Lower |
| Rare word rate | Words outside top-2000 frequency | Higher | Lower |
| Academic word rate | Words from Academic Word List | Higher | Lower |

**Important nuance on TTR:**
- Raw TTR is sensitive to text length (longer texts -> lower TTR mechanically)
- Use **MATTR** (Moving Average TTR with 50-word window) for length-normalized measurement
- Or use **MTLD** (Measure of Textual Lexical Diversity)

### 4.3 Sentence Structure Features

| Feature | What it measures | Expected: Cheating | Expected: Genuine |
|---------|-----------------|--------------------|--------------------|
| Avg sentence length | Syntactic complexity | Longer | Shorter/fragments |
| Sentence length std | Consistency | Lower (uniformly complex) | Higher (variable) |
| Subordination ratio | Clause embedding | Higher | Lower |
| Fragment rate | Incomplete sentences | Lower | Higher |
| Conjunction rate | Sentence linking | Higher | Lower |

### 4.4 Spontaneity Markers (Text-Based)

| Feature | What it measures | Expected: Cheating | Expected: Genuine |
|---------|-----------------|--------------------|--------------------|
| Self-reference rate | "I", "my", "me", "myself" | Lower | Higher |
| Hedging rate | "I think", "maybe", "kind of" | Lower | Higher |
| Discourse markers | "you know", "like", "basically" | Lower | Higher |
| Filler transcription rate | "um", "uh" in transcript | Near zero | Present |
| Repetition rate | Repeated ideas/phrases | Lower | Higher |
| Personal anecdote markers | "in my experience", "I remember" | Lower | Higher |

### 4.5 AI-Generated Text Detection on Transcripts

**Perplexity-based approach:**
- AI-generated text has **lower perplexity** (more predictable word sequences)
- Human spontaneous speech transcripts have **higher perplexity** (unexpected word choices, incomplete thoughts)
- Best detectors achieve ~89% accuracy on written text using perplexity + GLTR features
- However: detection degrades with newer models (GPT-4, Claude) and with spoken transcripts vs written text

**Practical approach:**
- Compute perplexity of transcript using a small LM (GPT-2, distilGPT-2)
- Low perplexity + high vocabulary complexity = strong cheating signal
- **Caveat:** Spoken transcripts are inherently noisier than written text, so calibrate thresholds on your data

**Features:**
1. **Transcript perplexity** (via GPT-2 or similar)
2. **Average token log-probability** (how "expected" each word is)
3. **Top-k token overlap** (how often the actual word is in top-10 predictions)
4. **Burstiness** (variance in sentence-level perplexity -- AI text is more uniform)

### 4.6 Deception Detection Research (Adjacent Field)

**"Analysing Deception in Witness Memory through Linguistic Styles" (Brain Sciences, 2023)**

Key findings relevant to our task:
- **Truthful statements are LONGER** (more words, more sentences)
- Truthful statements have **more articulated sentence structures** and subordination
- Truthful statements contain **more cognitive criteria** (details, specific memories)
- Truthful speakers **admit memory gaps** more often
- Adjectives were the most-varying category (deceptive: 4.62%, truthful: 3.03%)

**"Acoustic-Prosodic and Lexical Cues to Deception" (TACL, 2020)**
- Combining prosodic + lexical + acoustic features outperforms any single modality
- Late fusion of separate classifiers works better than early feature concatenation

**"Cognitive Load as Key to Lie Detection" (2025, Fallah et al.)**
- Filled pauses ("um") occur **less frequently** and are **shorter** during lying
- Pause patterns reflect cognitive load differences
- Spontaneous truthful speech has more natural hesitation patterns

**Relevance:** While our task isn't deception detection per se, the reading-from-prepared-material scenario has similar cognitive load characteristics to rehearsed/prepared statements.

---

## 5. Production Systems for Interview/Exam Fraud Detection

### 5.1 Talview (Anti-Parakeet AI)

**What they detect:**
- Background voices, whispering, AI-generated coaching
- "Scripted timing" in response delivery
- Device detection (phones, earpieces)
- Behavioral analysis of response patterns

**Technical approach:** Multi-layered: app blocking + audio intelligence + behavioral monitoring
**Accuracy:** No published metrics
**Limitation:** Primarily focused on detecting external audio sources, not reading from screen

### 5.2 Sherlock AI

**What they detect:**
- Background application activity
- Response behavior and interaction patterns
- "How answers are formed, how quickly they appear"
- Changes in interaction flow when external help is present

**Technical approach:** Real-time monitoring + AI-driven analysis
**Accuracy:** No published metrics
**Limitation:** Minimal technical specifics published; marketing-heavy

### 5.3 Mercer Mettl (Your Company)

**Current audio proctoring capabilities:**
- Speech Sense: differentiates human speech from background noise
- Counts unique speakers in audio feed
- Detects whispers, low sounds, mumbling
- Claims "95% accuracy" for AI-based proctoring

**Gap:** Currently focused on **who is speaking** (multiple speakers, external voices), NOT on **how/what they are speaking** (prepared vs spontaneous content). Your project fills this gap.

### 5.4 Hyring / Other Proctoring Platforms

- Flag "unnaturally fast" or "overly polished" responses
- Some use response-time analysis (how quickly after question does answer start)
- Most rely on video-based signals (eye movement, head tracking) more than audio analysis

**Key industry gap:** No production system published combines linguistic analysis of speech content with acoustic spontaneity detection. This is an open opportunity.

---

## 6. Whisper Variants for Feature Extraction

### 6.1 CrisperWhisper (RECOMMENDED)

**Why it's best for our pipeline:**
- Verbatim transcription preserving ALL disfluencies (fillers, false starts, repairs)
- Word-level timestamps with 84.7% F1 (0.2s collar)
- Filler detection near-perfect
- Noise robust (79.5% F1 with noise vs WhisperX's 59.0%)
- No additional model needed (unlike WhisperX which needs wav2vec2 for alignment)
- Available: `nyrahealth/CrisperWhisper` on HuggingFace, also `faster_CrisperWhisper`

**Benchmark results:**
| Dataset | Standard Whisper WER | CrisperWhisper WER |
|---------|---------------------|--------------------|
| AMI Meeting Corpus | 16.82% | **9.72%** |
| TED-LIUM | 4.01% | **3.26%** |
| LibriSpeech (read) | Same | Same |

**Practical notes:**
- Based on Whisper Large v3, so requires GPU or patience on CPU
- `faster_CrisperWhisper` variant available for speed
- Filters tokens shorter than 50ms to mitigate hallucinations
- Pause heuristic caps durations at 160ms for word boundaries

### 6.2 Whisper Encoder Embeddings for Classification

**Architecture for paralinguistic tasks:**
- Whisper encoder processes log-Mel spectrogram into embeddings
- Embeddings capture both temporal and spectral relationships
- Can fine-tune encoder with LoRA or use frozen + classifier head
- SpeechBrain provides easy integration for downstream tasks

**Layer analysis (applies to all transformer speech models):**
| Layer Region | Encodes | Relevance to our task |
|---|---|---|
| Bottom (1-3) | Acoustic features, pitch, timbre | Prosodic signal |
| Middle (4-8) | Paralinguistic traits, speaker characteristics | Speaking style |
| Top (9-12) | Linguistic content, phonemic structure | Vocabulary, syntax |

**Limitation:** Whisper was optimized for linguistic content (ASR), so it's weaker for speaker identification and pure paralinguistic tasks compared to WavLM/HuBERT. But for our task, we NEED both linguistic and acoustic, making Whisper ideal.

---

## 7. Speech Rhythm Metrics

### 7.1 Pairwise Variability Index (PVI)

**nPVI (normalized) and rPVI (raw):**
- Measure variability in consecutive vowel/consonant interval durations
- Achieved **91.3% correct classification** between speech styles in one study
- Spontaneous speech has **higher normalized PVI** than read speech (more rhythmic variability)

**Limitations:**
- Very sensitive to elicitation method and syllable complexity
- Large between-speaker variation
- Language-dependent (mora-timed languages like Japanese behave differently)

### 7.2 Speaking Rate Features

**Research findings:**
- Speaking rate in spontaneous speech: ~256 syllables/min
- Speaking rate in reading: ~235 syllables/min
- Articulation rate in speaking: ~360 syllables/min
- Articulation rate in reading: ~311 syllables/min
- **Spontaneous speech is faster but with more pauses** (net speaking rate may be similar)

**Features to extract:**
1. Speaking rate (syllables/sec including pauses)
2. Articulation rate (syllables/sec excluding pauses)
3. Speaking rate variability (std across 5s windows)
4. Articulation ratio (phonation time / total time)
5. Pause-to-speech ratio

---

## 8. SUPERB Benchmark and Self-Supervised Models

### 8.1 SUPERB Results

SUPERB evaluates self-supervised speech models across 10 tasks. Relevant findings:
- HuBERT Large achieves best overall performance (2.94% ASR WER)
- WavLM-Base+ is best for paralinguistic tasks (emotion, speaker traits)
- All SSL models vastly outperform traditional features (FBANK)

### 8.2 Best Models for Our Task

| Model | Strengths | Weaknesses | Best for |
|-------|-----------|------------|----------|
| **CrisperWhisper** | Verbatim ASR + timestamps + fillers | Large, no paralinguistic optimization | Transcription pipeline |
| **WavLM-Base+** | Best paralinguistic encoding | No linguistic content awareness | Acoustic features only |
| **Whisper encoder** | Both acoustic + linguistic | Weaker paralinguistic than WavLM | End-to-end classification |
| **HuBERT** | Strong general representations | Superseded by WavLM for our use | General baseline |

### 8.3 TRILLSSON (Distilled Paralinguistic Model)

**"Distilled Universal Paralinguistic Speech Representations" (Google, 2022)**
- Distills large SSL models into small, efficient models for paralinguistic tasks
- Designed for on-device inference
- Could be useful if we need very fast acoustic feature extraction

---

## 9. Recommended Multi-Signal Architecture

### 9.1 The Full Pipeline

```
Audio (50-60s clip)
    |
    +-- CrisperWhisper ------> Verbatim Transcript + Word Timestamps
    |                               |
    |                    +----------+----------+
    |                    |                     |
    |              Text Features          Pause Features
    |              (Section 4)            (Section 2)
    |                    |                     |
    +-- openSMILE -------> Acoustic Features   |
    |   (eGeMAPS)         (88 features)        |
    |                          |               |
    +-- Pitch extraction ----> Prosodic Features|
        (librosa/PRAAT)       (Section 1.3)    |
                                   |           |
                    +--------------+-----------+
                    |
              Feature Vector (~150-200 features)
                    |
              XGBoost/LightGBM
                    |
              [Cheating / Not Cheating / Uncertain]
```

### 9.2 Feature Groups with Priority

**Priority 1 (implement first, strongest signal):**
1. Filler/disfluency rate from CrisperWhisper transcript
2. Type-Token Ratio (MATTR)
3. Complex word rate
4. Self-reference rate ("I", "my")
5. Discourse marker rate
6. Repetition rate (repeated n-grams)
7. Average sentence length + std

**Priority 2 (add next, strong acoustic signal):**
8. Pause-before-content-word ratio (needs POS tagging + timestamps)
9. Mid-phrase pause rate
10. Pause duration distribution (mean, std, skewness)
11. Speaking rate variability
12. Articulation ratio

**Priority 3 (add for refinement):**
13. F0 mean, std, range, skewness
14. F0 declination slope
15. eGeMAPS feature set (88 features)
16. Transcript perplexity (GPT-2)
17. nPVI rhythm metrics

**Priority 4 (optional, diminishing returns):**
18. Residual embeddings (WavLM - text)
19. Whisper encoder embedding + neural classifier
20. Response latency (time from question to first word)

### 9.3 Expected Performance by Approach

| Approach | Expected Precision | Expected Recall | Effort |
|----------|-------------------|-----------------|--------|
| Text features only (Priority 1) | 70-80% | 65-75% | 1-2 days |
| Text + Pause features (P1 + P2) | 78-88% | 72-82% | 3-4 days |
| Full multi-signal (P1-P3) | 82-92% | 78-85% | 5-7 days |
| + Neural (P4) | 85-93% | 80-87% | 10+ days |

### 9.4 Production Deployment Considerations

| Component | Size | CPU Latency (per file) | GPU Latency |
|-----------|------|----------------------|-------------|
| CrisperWhisper (large) | ~3GB | 30-60s for 50s audio | 3-5s |
| faster_CrisperWhisper | ~1.5GB | 15-30s | 2-3s |
| Whisper-small (fallback) | ~200MB | 10-15s | 1-2s |
| spaCy POS tagger | ~50MB | <0.1s | - |
| openSMILE eGeMAPS | ~10MB | <0.5s | - |
| XGBoost classifier | ~1MB | <0.01s | - |
| GPT-2 perplexity | ~500MB | 1-2s | <0.5s |
| **Total (with CrisperWhisper)** | **~3.5GB** | **~35-65s** | **~5-8s** |
| **Total (with Whisper-small)** | **~750MB** | **~12-18s** | **~2-3s** |

**Recommendation:**
- If GPU available: CrisperWhisper for best disfluency/timestamp quality
- If CPU-only laptop: Whisper-small with whisper-timestamped for word timestamps, then use text features + approximate pause features
- Key trade-off: CrisperWhisper gives you verbatim disfluencies (critical for filler rate), Whisper-small may miss them

### 9.5 Fallback for CPU-Only Deployment

If CrisperWhisper is too slow on CPU:
1. Use **Whisper-small** for transcription (fast, good WER)
2. Use **whisper-timestamped** for word-level timestamps (no extra model)
3. **Manually detect fillers** via acoustic analysis (energy dips + short voiced segments between pauses)
4. Text features remain fully available regardless of Whisper variant

---

## 10. Key Research Gaps and Open Questions

1. **No published system** combines linguistic transcript analysis with acoustic spontaneity detection for exam proctoring -- this is novel territory.

2. **Accented English** impacts both ASR quality and prosodic features. Need to evaluate on Indian-accented English specifically.

3. **Perplexity-based AI detection on spoken transcripts** is unstudied. Spoken transcripts have natural disfluencies that may confound detectors trained on written text.

4. **Pause-before-content-word analysis** has strong theoretical backing but no published production system uses it. Need to validate that CrisperWhisper timestamps are accurate enough for this measurement.

5. **Calibration threshold** for "uncertain" class needs careful tuning on company data to balance false positive cost vs missed cheaters.

---

## 11. Immediate Next Steps

1. **Install CrisperWhisper** and test on a few company files to assess:
   - Transcription quality on accented English
   - Whether fillers are preserved
   - Word-level timestamp accuracy for pause measurement

2. **Build text feature extractor** (Priority 1 features) and test on company labeled data

3. **Build pause feature extractor** using CrisperWhisper timestamps + spaCy POS tags

4. **Train XGBoost** on combined features, evaluate precision/recall

5. **Error analysis** on misclassified files to identify which features need refinement

---

## 12. References

### Core Papers
1. [Classification of Spontaneous and Scripted Speech for Multilingual Audio](https://arxiv.org/abs/2412.11896) - Elisha et al., SLT 2024
2. [CrisperWhisper: Accurate Timestamps on Verbatim Speech Transcriptions](https://arxiv.org/abs/2408.16589) - Zusag et al., Interspeech 2024
3. [Residual Speech Embeddings for Tone Classification](https://arxiv.org/html/2502.19387v1) - Feb 2025
4. [Can You Tell Apart Spontaneous and Read Speech if You Just Look at Prosody?](https://link.springer.com/chapter/10.1007/978-3-642-57745-1_47) - Batliner et al.
5. [Structure of Pauses in Speech in Context of Speaker Verification and Classification of Speech Type](https://link.springer.com/article/10.1186/s13636-016-0096-7) - Cernak et al., EURASIP 2016

### Disfluency and Hesitation
6. [Hesitations in Spontaneous Speech: Acoustic Analysis and Detection](https://link.springer.com/chapter/10.1007/978-3-319-66429-3_39)
7. [Audio and ASR-based Filled Pause Detection](https://sail.usc.edu/publications/files/Chatziagapi-ACII2022.pdf) - USC SAIL
8. [Automatic Disfluency Detection from Untranscribed Speech](https://www.microsoft.com/applied-sciences/uploads/publications/134/automatic-disfluency-detection.pdf) - Microsoft Research

### Deception and Cognitive Load
9. [Analysing Deception in Witness Memory through Linguistic Styles](https://pmc.ncbi.nlm.nih.gov/articles/PMC9953826/) - Brain Sciences 2023
10. [Acoustic-Prosodic and Lexical Cues to Deception and Trust](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00311/43547/) - TACL 2020
11. [Cognitive Load as Key to Lie Detection](https://journals.sagepub.com/doi/10.1177/00332941251352448) - Fallah et al. 2025

### Pause and Rhythm Analysis
12. [The Significance of Pauses in Spontaneous Speech](https://link.springer.com/article/10.1007/BF01067111) - Journal of Psycholinguistic Research
13. [The Role of Fluency in Taking Pauses while Reading Aloud and During Spontaneous Speech](https://www.researchgate.net/publication/376276373)
14. [Applying Rhythm Metrics to Non-native Spontaneous Speech](https://www.cstr.ed.ac.uk/downloads/publications/2013/laiEtAl2012rhythm.pdf)

### Models and Benchmarks
15. [SUPERB: Speech Processing Universal PERformance Benchmark](https://www.alphaxiv.org/overview/2105.01051v4)
16. [TRILLSSON: Distilled Universal Paralinguistic Speech Representations](https://arxiv.org/pdf/2203.00236)
17. [A Fine-tuned Wav2vec 2.0/HuBERT Benchmark For Speech Emotion Recognition](https://arxiv.org/abs/2111.02735)
18. [Whisper Has an Internal Word Aligner](https://arxiv.org/html/2509.09987v1) - 2025

### Tools
19. [WhisperX: Word-level Timestamps and Diarization](https://github.com/m-bain/whisperX)
20. [CrisperWhisper GitHub](https://github.com/nyrahealth/CrisperWhisper)
21. [openSMILE: Audio Feature Extraction](https://www.audeering.com/research/opensmile/)

### Industry / Proctoring
22. [Talview: Stop Parakeet AI Cheating](https://www.talview.com/en/stop-parakeet-ai-cheating)
23. [Sherlock AI: Detect and Prevent Parakeet AI](https://www.withsherlock.ai/blog/detect-and-prevent-parakeet-ai)
24. [Mercer Mettl Audio Proctoring](https://mettl.com/glossary/a/audio-proctoring/)
25. [Veritext: AI-generated Text Detection Based on Perplexity and GLTR](https://www.sciencedirect.com/science/article/pii/S1877050925027838)
