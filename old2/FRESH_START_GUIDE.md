# Cheating Detection in Speech: Fresh Start Guide
# Complete Research, Revised Approaches, and Implementation Plan

---

## Table of Contents
1. [Problem Reframing: This Is NOT Read vs Spontaneous](#1-problem-reframing)
2. [Why Previous Approaches Failed](#2-why-previous-approaches-failed)
3. [Key Research Findings](#3-key-research-findings)
4. [The Two Signals You Actually Need](#4-the-two-signals)
5. [Ranked Approaches (Best to Worst)](#5-ranked-approaches)
6. [Recommended Architecture](#6-recommended-architecture)
7. [Data Strategy](#7-data-strategy)
8. [Training Plan](#8-training-plan)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Deployment Pipeline](#10-deployment-pipeline)
11. [Experimental Roadmap](#11-experimental-roadmap)
12. [References](#12-references)

---

## 1. Problem Reframing: This Is NOT Read vs Spontaneous {#1-problem-reframing}

### The Real Task
You're not classifying "read speech vs spontaneous speech." You're detecting **whether a candidate is using prepared/AI-generated answers in an assessment**. This is a cheating detection problem, not a speaking style problem.

The distinction matters because:
- A "read vs spontaneous" model tries to detect HOW someone speaks (prosody, hesitation, fluency)
- A "cheating detection" model should detect BOTH how AND what someone speaks (prosody + linguistic complexity + vocabulary)

### Evidence From Your Own Observations
You noticed two key signals while listening to misclassified files:
1. **Cheaters**: Complex vocabulary, complex sentence structure (linguistic signal)
2. **Non-cheaters**: Common vocabulary, repeat themselves, repeat their points even when fluent (linguistic signal)

These are **content-level signals**, not acoustic ones. Your Wav2Vec2/WavLM models are literally incapable of detecting these -- they encode audio patterns, not word meanings.

### The Correct Problem Statement
> Given a 50-60 second audio clip of a candidate answering an assessment question, classify whether the candidate is:
> - **Cheating**: Reading from a prepared source (AI answer, notes, screen) -- complex vocabulary, structured sentences, unnatural fluency
> - **Not cheating**: Genuinely answering -- common vocabulary, natural repetition, self-correction, thinking pauses
> - **Uncertain**: Ambiguous cases flagged for human review

---

## 2. Why Previous Approaches Failed {#2-why-previous-approaches-failed}

### AllStar Data Teaches the Wrong Thing
AllStar training data has a clean, artificial correlation:
- Read = fluent, no hesitation
- Spontaneous = disfluent, lots of hesitation

Company data breaks this correlation completely:
- Cheaters reading AI answers can still stumble and hesitate
- Honest candidates can be articulate and fluent
- Non-native speakers add noise to both categories

**Result on 288 company audios: 45% precision / 60% recall** -- worse than random.

### GigaSpeech Makes It Worse
- Dataset quality issues (speaker switching within clips)
- Audiobooks vs podcasts is a completely different distinction than cheating vs honest
- Added more volume of the wrong training signal
- Result: 62% precision for Wav2Vec2, 50% for Whisper

### Pure Acoustic Models Are Fundamentally Limited for This Task
The strongest signal (vocabulary complexity, sentence structure) is invisible to:
- Wav2Vec2: Encodes acoustic/phonetic patterns, not word meanings
- WavLM: Better paralinguistic encoding, but still no semantic understanding
- Raw prosodic features: Pitch, rhythm, pauses -- no content awareness

**Only Whisper has access to linguistic content** because it was trained for ASR. Its encoder representations encode both acoustic AND linguistic information. But even Whisper frozen + MLP only gave 50% precision on GigaSpeech-trained model.

### The 10-Second Window Problem
Longer windows (10s) performed WORSE (46% precision, 50% recall) because:
- They span topic transitions within a single answer
- A cheater might read for 5s then think for 5s -- the 10s window averages to "uncertain"
- Short windows (5s) at least catch clean read/spontaneous segments

---

## 3. Key Research Findings {#3-key-research-findings}

### The Definitive Paper (December 2024)
**"Classification of Spontaneous and Scripted Speech for Multilingual Audio"** (arXiv:2412.11896)
- Tested on ~4,000 Spotify podcasts across 15 languages
- **Whisper frozen encoder achieves 0.95 AUC, generalizes cross-domain to 0.92-0.95**
- Handcrafted acoustic features collapse cross-domain (0.87 -> 0.46)
- Fine-tuning Whisper didn't improve over frozen -- representations are already good

### BUT: This Paper's Task Is Different From Yours
Their task: podcast host reading ad copy vs having conversation (clean acoustic distinction).
Your task: candidate reading AI-generated answer vs genuinely answering (mixed acoustic/linguistic distinction).

**Whisper's linguistic encoding is still your best bet**, but you need to train on YOUR data, not AllStar.

### Prosody Alone = 55-65% Accuracy (Batliner et al.)
For speaker-independent classification using only prosodic features. This is essentially what your Wav2Vec2 models are doing -- learning prosodic patterns. On your data it's performing at that level (45-60%).

### Residual Embeddings (arXiv:2502.19387, Feb 2025)
Novel technique: separate "what was said" from "how it was said" by regressing speech embeddings on text embeddings and using the residual.
- WavLM residual embeddings: 98-100% on tone classification
- Could be powerful for your task, but adds pipeline complexity (need ASR first)

### Self-Supervised Model Layers
| Layer Region | What It Encodes |
|---|---|
| Bottom (1-3) | Acoustic features, pitch, timbre |
| Middle (4-8) | Paralinguistic traits, speaker characteristics |
| Top (9-12) | Linguistic content, phonemic structure |

For your task, **all layers matter** because you need both prosody AND linguistic content.

---

## 4. The Two Signals You Actually Need {#4-the-two-signals}

### Signal 1: Acoustic/Prosodic (How They Speak)
- Pitch variability (reading = more monotone or unnaturally varied)
- Pause patterns (reading = pauses at wrong places, or unnaturally even spacing)
- Speaking rate consistency (reading = more constant rate)
- Hesitation patterns (honest = "um", "uh", false starts, self-correction)
- Breath timing (reading = unnatural breath placement)

**Captured by**: Wav2Vec2, WavLM, Whisper encoder (all layers)
**Limitation**: Alone gives ~45-65% on your data

### Signal 2: Linguistic/Content (What They Say)
- **Vocabulary complexity**: Cheaters use words like "comprehensive", "multifaceted", "leveraging" -- AI-generated text fingerprint
- **Sentence structure**: Cheaters produce complete, grammatically complex sentences. Honest candidates use fragments, restart sentences
- **Repetition**: Honest candidates repeat ideas in different words. Cheaters don't.
- **Self-reference**: Honest candidates say "I think", "in my experience", "like". Cheaters produce impersonal, essay-like text.
- **Discourse markers**: "you know", "basically", "so like" = honest. Absence of these = suspicious.

**Captured by**: Whisper ASR transcription -> text analysis, OR Whisper encoder top layers
**Limitation**: Needs good ASR quality (Whisper is solid for this)

### The Winning Combination
You need BOTH signals. Three realistic ways to combine them:

| Method | Acoustic | Linguistic | Complexity | Expected Performance |
|---|---|---|---|---|
| **A: Whisper encoder (all layers) + classifier** | Yes (bottom layers) | Partial (top layers) | Low | Good |
| **B: Whisper ASR -> text features + acoustic features** | Via separate model | Yes (full text) | Medium | Better |
| **C: Whisper ASR -> text features + Whisper encoder -> acoustic features** | Yes | Yes | Medium-High | Best |

---

## 5. Ranked Approaches (Best to Worst) {#5-ranked-approaches}

### TIER 1: Most Likely to Work

#### Approach 1: Whisper ASR + Text Features + Simple Acoustic Features (RECOMMENDED FIRST)
**Why this is #1**: Your strongest signal is LINGUISTIC (vocabulary, sentence structure). Whisper can transcribe the audio, then you analyze the text. This is simple, interpretable, and directly targets what you observed.

**Architecture:**
```
Audio (50-60s) -> Whisper ASR -> Transcript text
                                      |
                              Text Feature Extraction:
                              - Type-Token Ratio (vocabulary diversity)
                              - Average word length
                              - % complex words (3+ syllables)
                              - Sentence length mean/std
                              - Filler word rate ("um","uh","like","you know")
                              - Repetition rate (repeated n-grams)
                              - Self-reference rate ("I","my","me")
                              - Discourse marker rate
                              - Named entity density
                              - Perplexity (via small LM -- how "AI-like" is the text)

Audio -> Acoustic Feature Extraction:
                              - Pitch mean/std/range
                              - Speaking rate + variability
                              - Pause count, duration, placement
                              - Energy variability
                              - Hesitation rate (filled pauses)

All features -> XGBoost/LightGBM classifier -> cheating/not_cheating
```

**Pros:**
- Directly captures your strongest signal (vocabulary complexity)
- Interpretable -- you can show WHY a candidate was flagged ("used vocabulary complexity score of X, zero filler words, no repetition")
- Simple to deploy (Whisper + feature extraction + tree model)
- Text features are domain-invariant (complex vocab is complex vocab regardless of recording quality)
- Can explain decisions to managers and candidates

**Cons:**
- Whisper ASR quality on accented speech may vary
- Need Whisper model for inference (larger deployment)
- Two-stage pipeline (ASR then classification)

**Expected performance:** 75-85% precision, 70-80% recall (major improvement over 45%)
**Model size:** Whisper-Small (~200MB) + XGBoost (~1MB)
**Effort:** 3-4 days to implement and test

#### Approach 2: Whisper Encoder (All Layers) + Learnable Layer Weights + Company Fine-Tuning
**Why this is #2**: If Approach 1 works, this is the neural version that may squeeze out more performance by learning features end-to-end.

**Architecture:**
```
Audio -> Whisper Feature Extractor (log-mel)
      -> Whisper Encoder (frozen or LoRA on top layers)
      -> Learnable weighted sum of ALL encoder layers
      -> Classifier: Linear(dim->256) -> GELU -> Dropout -> Linear(256->3)
                                                              [cheat / not_cheat / uncertain]
```

**Key difference from your previous Whisper training:**
- Train on COMPANY DATA, not AllStar
- Use ALL encoder layers (weighted), not just last hidden state
- LoRA on top layers to adapt to your domain
- 3-class output (cheat/not_cheat/uncertain) instead of binary

**Expected performance:** 80-90% precision, 75-85% recall
**Effort:** 3-4 days

#### Approach 3: Dual-Branch (Text Features + Whisper Encoder Fusion)
**Architecture:**
```
Audio -> Whisper ASR -> Text -> Text features (Branch A)
Audio -> Whisper Encoder -> Acoustic embedding (Branch B)

Concatenate [Branch A, Branch B] -> Classifier
```

**Best theoretical performance but most complex. Try this if Approaches 1-2 plateau.**

### TIER 2: Worth Trying If Tier 1 Hits a Wall

#### Approach 4: WavLM-Base+ Trained on Company Data Only
- Drop AllStar entirely
- Train directly on 500 company files (400 train / 100 test)
- WavLM has the best paralinguistic encoding
- Won't capture linguistic content but might learn company-specific patterns
- Expected: 65-75% precision (better than 45% but limited by no linguistic signal)

#### Approach 5: Multi-Feature Ensemble
- Whisper ASR text features (XGBoost)
- WavLM acoustic embedding (neural)
- Prosodic features from openSMILE/eGeMAPS
- Stacking ensemble (logistic regression on top)
- High effort, diminishing returns

### TIER 3: Quick Wins (Do Regardless)

#### Approach 6: Text-Only Baseline (Whisper ASR -> Text Complexity Score)
- **Do this FIRST as a sanity check** -- takes 1 day
- Run Whisper on your 500 company files
- Extract just: type-token ratio, filler word rate, avg sentence length, repetition rate
- Train logistic regression
- If this alone gets >60% accuracy, the linguistic signal is confirmed and Approach 1 is the way

#### Approach 7: Threshold + TTA on Existing Model
- Apply TTA (3 window offsets) to existing Wav2Vec2 model
- Calibrate threshold on the 288 labeled files
- Quick ~2 hour experiment
- Won't fix the fundamental problem but might give a few % improvement

---

## 6. Recommended Architecture: The Build Plan {#6-recommended-architecture}

### Step-by-Step Implementation Order

#### Step 1: Text-Only Baseline (Day 1) -- VALIDATES THE HYPOTHESIS
```python
# Pipeline:
# 1. Run Whisper-Small on all 500 company files -> get transcripts
# 2. Extract text features from each transcript
# 3. Train LogisticRegression / XGBoost
# 4. If accuracy > 60%, linguistic signal is real -> proceed to full approach

text_features = {
    "type_token_ratio": n_unique_words / n_total_words,
    "avg_word_length": mean(len(word) for word in words),
    "filler_rate": count(um|uh|hmm|like|you know|basically) / n_total_words,
    "repetition_rate": count(repeated_bigrams) / n_total_bigrams,
    "self_reference_rate": count(I|my|me|myself) / n_total_words,
    "avg_sentence_length": mean(words_per_sentence),
    "sentence_length_std": std(words_per_sentence),
    "complex_word_rate": count(words_with_3plus_syllables) / n_total_words,
    "discourse_marker_rate": count(so|well|anyway|actually) / n_total_words,
}
```

#### Step 2: Full Acoustic + Text Pipeline (Days 2-4)
If text baseline works, build the full Approach 1 pipeline.

#### Step 3: Neural Approach (Days 5-7)
If Approach 1 performance is good, try Approach 2 (Whisper encoder fine-tuning) to see if end-to-end learning helps.

#### Step 4: Company Fine-Tuning on EC2 (Days 7-10)
Train final model on EC2 with company data.

### Key Architecture Decisions

**Why XGBoost/LightGBM for the classifier (not neural)?**
- 500 files is a SMALL dataset for neural training
- Tree models handle small datasets better and don't overfit as easily
- Interpretable: you get feature importance (critical for explaining flagged candidates)
- Fast inference: tree models are nearly instant on CPU
- You already have XGBoost experience in the project

**Why Whisper-Small over Whisper-Medium?**
- 244M vs 769M parameters
- ~200MB vs ~750MB quantized ONNX
- Whisper-Small has excellent English ASR quality (WER ~5-8%)
- For accented English: Whisper-Small may struggle more, but still workable
- Try Small first, upgrade to Medium only if ASR quality is the bottleneck

**Why 3-class instead of 2-class?**
- Add "uncertain" class for borderline cases (e.g., someone with naturally complex vocabulary)
- Route "uncertain" to human review
- Reduces false positives while maintaining recall on clear cases
- Train: use 2-class labels + mark predictions with confidence 0.4-0.6 as "uncertain" at inference

---

## 7. Data Strategy {#7-data-strategy}

### What You Have: 500 Company Files
- 219 concentrated cheater files (mixed with non-cheaters)
- 288 files with clear labels
- ~50s average duration, 1min max
- File-level labels: cheating / not cheating

### Recommended Split
| Set | Size | Purpose |
|---|---|---|
| Train | 350 files (70%) | Model training |
| Validation | 75 files (15%) | Hyperparameter tuning, early stopping |
| Test | 75 files (15%) | Final evaluation ONLY (never touch during development) |

**Important:** Stratify by label AND by accent/language if possible.

### Do You Need Segment-Level Timestamps?
**No, not for the recommended approach.** Here's why:
- Your goal is file-level flagging (cheating/not cheating)
- Approach 1 (text features) operates on the full transcript -- no windowing needed
- Approach 2 (Whisper encoder) can use file-level labels with pooling
- Segment timestamps would help for Approach 2's window-level training, but you can start without them

**Save the manual annotation effort for later**, only if file-level classification hits a ceiling and you need segment-level granularity.

### Should You Use AllStar Data?
**Probably not for primary training.** AllStar teaches:
- "Fluent reading" = read (but your cheaters can be disfluent)
- "Hesitant speech" = spontaneous (but your honest candidates can be fluent)

**Possible limited use:** Pre-train on AllStar, then fine-tune on company data. But with 500 company files, you likely don't need pre-training for a tree-based model.

### Data Augmentation
For text features (Approach 1): No augmentation needed -- text features are robust.

For audio features (Approach 2):
```python
augmentations = {
    "speed_perturbation": [0.9, 0.95, 1.0, 1.05, 1.1],
    "noise_addition": {"snr_db": [15, 20, 30]},
    "pitch_shift": {"semitones": [-1, 0, 1]},
}
```

---

## 8. Training Plan {#8-training-plan}

### Phase 1: Validate Linguistic Signal (Day 1, Local Machine)

```bash
# Step 1: Transcribe all 500 files with Whisper-Small
# Step 2: Extract text features
# Step 3: Train LogisticRegression + XGBoost
# Step 4: Report accuracy
```

If text-only accuracy > 60%: linguistic signal confirmed, proceed.
If text-only accuracy < 55%: signal is weak, pivot to Approach 2 (pure neural).

### Phase 2: Full Pipeline (Days 2-4, Local Machine)

```bash
# Step 1: Extract acoustic features (pitch, rate, pauses, energy)
# Step 2: Combine text + acoustic features
# Step 3: Train XGBoost with hyperparameter search
# Step 4: Evaluate on held-out test set
```

### Phase 3: Neural Approach (Days 5-7, EC2 with Company Data)

```bash
# Step 1: Fine-tune Whisper encoder with LoRA on company data
# Step 2: Compare with XGBoost approach
# Step 3: Pick best approach
# Step 4: Export ONNX
```

### Phase 4: Production (Days 8-10)

```bash
# Step 1: Final model selection
# Step 2: ONNX export + INT8 quantization
# Step 3: Update predict_cpu.py
# Step 4: Validate on test set one final time
# Step 5: Package for deployment
```

---

## 9. Evaluation Framework {#9-evaluation-framework}

### Primary Metrics
- **Precision** (most important for your use case): Flagging an honest candidate as cheater is worse than missing a cheater
- **Recall**: Catch as many cheaters as possible
- **Target**: Precision >= 80%, Recall >= 70% (realistic given the difficulty)

### How to Evaluate
```
For each audio file:
  1. Run pipeline -> get prediction (cheat / not_cheat / uncertain)
  2. Compare with ground truth label
  3. Compute precision, recall, F1

Additionally:
  - Precision @ K: What % of the top K most confident "cheating" flags are correct?
  - Uncertain rate: What % of files go to human review?
  - Per-accent breakdown: Does the model discriminate against accents?
```

### Error Analysis Protocol
For every misclassified file:
1. Listen to the audio
2. Read the Whisper transcript
3. Check which features were dominant in the prediction
4. Categorize: was it an acoustic error, linguistic error, or labeling error?

This will tell you whether to invest more in acoustic or linguistic features.

---

## 10. Deployment Pipeline {#10-deployment-pipeline}

### Approach 1 (Text + Acoustic Features)
```
Company Laptop:
  - Whisper-Small ONNX (~200MB) for transcription
  - Feature extraction code (pure Python, no ML deps)
  - XGBoost model (~1MB)
  - Total: ~201MB, runs on any CPU

Inference time per file: ~10-15s (mostly Whisper transcription)
```

### Approach 2 (Whisper Encoder Neural)
```
Company Laptop:
  - Whisper-Small ONNX quantized (~200MB)
  - runs on any CPU

Inference time per file: ~5-10s
```

---

## 11. Experimental Roadmap {#11-experimental-roadmap}

```
Day 1: TEXT BASELINE (Critical Experiment)
  |
  Run Whisper-Small on 500 company files
  Extract text features
  Train LogisticRegression + XGBoost
  |
  ├── If accuracy > 60%: Linguistic signal CONFIRMED
  |     |
  |     Day 2-3: Add acoustic features (pitch, pauses, rate)
  |     Day 3-4: Optimize XGBoost, evaluate on test set
  |     Day 5: Try Whisper encoder approach on EC2
  |     Day 6-7: Compare, pick winner
  |     Day 8-10: ONNX export, deployment prep
  |
  └── If accuracy < 55%: Linguistic signal WEAK
        |
        Day 2-3: Go straight to Whisper encoder + LoRA on EC2
        Day 4-5: Try WavLM-Base+ on company data
        Day 6-7: Ensemble if needed
        Day 8-10: ONNX export, deployment prep
```

### Milestones
| Day | Milestone | Success Criteria |
|---|---|---|
| 1 | Text baseline accuracy | > 60% = proceed |
| 4 | Full pipeline on test set | Precision > 70% |
| 7 | Final model selected | Precision > 80%, Recall > 70% |
| 10 | ONNX deployed | Runs on laptop, < 15s per file |

---

## 12. References {#12-references}

### Directly Relevant
1. **arXiv:2412.11896** (Dec 2024) - "Classification of Spontaneous and Scripted Speech for Multilingual Audio" - Whisper encoder 0.95 AUC, best cross-domain generalization
2. **arXiv:2306.08012** (2023) - Read/spontaneous classification on ALLSSTAR corpus, 88% accuracy
3. **arXiv:2502.19387** (Feb 2025) - Residual speech embeddings, disentangle content from style
4. **Batliner et al. (2000)** - Prosody alone: 55-65% speaker-independent accuracy

### Model Architecture
5. **arXiv:2107.04734** (Pasad et al. 2021) - SSL model layer analysis
6. **arXiv:2210.07185** (2022) - SSL models for prosody tasks, WavLM best for pitch
7. **arXiv:2501.05310** (Chen et al. 2025) - Probing speaker attributes in SSL

### AI-Generated Text Detection (Adjacent Problem)
8. GPTZero, GLTR, DetectGPT approaches -- perplexity-based detection of AI text
9. Text complexity metrics for academic integrity
10. Stylometry techniques for authorship verification

---

## Quick Reference: What to Do Right Now

1. **DON'T** train any more models on AllStar data
2. **DO** run Whisper-Small transcription on your 500 company files (can do locally)
3. **DO** extract text features and train a simple classifier
4. **DO** analyze which text features are most predictive
5. **DO** keep the 500-file dataset clean and well-labeled
6. **LATER** consider neural approaches (Whisper encoder + LoRA) on EC2 with company data

---

## File Organization

```
Speech project/
├── old/                          # All previous code (preserved)
├── FRESH_START_GUIDE.md          # This document
├── project_context_for_claude_code.md  # Original context
├── 2676/                         # AllStar spontaneous (keep for reference)
├── 2677/                         # AllStar read (keep for reference)
├── data/                         # GigaSpeech data (can ignore)
├── src/                          # NEW code goes here
│   ├── transcribe.py             # Whisper ASR on company files
│   ├── text_features.py          # Text feature extraction
│   ├── acoustic_features.py      # Acoustic feature extraction
│   ├── train_classifier.py       # XGBoost/LightGBM training
│   ├── train_whisper_encoder.py  # Neural approach (Phase 3)
│   └── evaluate.py               # Evaluation + error analysis
├── predict_cpu.py                # Updated inference pipeline
├── export_onnx.py                # ONNX export for deployment
└── checkpoints/                  # New models
```
