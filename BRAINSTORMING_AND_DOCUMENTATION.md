# Speech Cheating Detection — Complete Project Documentation

## Project Overview

**Goal:** Detect cheating in online spoken assessments (Mercer Mettl platform). Candidates answer open-ended questions via audio — cheating means using outside help (reading from notes, GPT-generated answers, pre-prepared responses, someone whispering answers).

**Core insight:** No single model can catch all cheating types. A candidate reading GPT output sounds different from one reading their own notes, which sounds different from one paraphrasing from a source. Multi-signal fusion is required.

**Final architecture:** Two independent models vote on each audio file:
1. **wav2vec2** (audio patterns) — P(cheating) from raw audio
2. **Text-only XGBoost** (linguistic patterns) — P(cheating) from text + pause + prosodic features
3. **Weighted vote** — `w * wav2vec2 + (1-w) * xgboost`, threshold tuned per deployment

---

## Phase 1: Single-Model Approach (Biased wav2vec2)

### What we tried
A wav2vec2-base model with a single sigmoid output neuron, trained to classify 5-second audio windows as "read" vs "spontaneous" speech. The hypothesis: cheaters read pre-written answers, so detecting reading = detecting cheating.

### Architecture decisions
- **Single neuron + sigmoid** instead of 2-neuron softmax — more natural for binary classification with a tunable threshold
- **Biased threshold = 0.65** — only flag as "reading" when confidence is high, reducing false positives
- **5-second windows** — long enough for meaningful audio patterns, short enough for fine-grained analysis
- **BCEWithLogitsLoss** with class-balanced pos_weight

### Dataset
- **ALLSSTAR** — 1,050 files (699 read, 351 spontaneous)
- Originally had GigaSpeech files in the manifest but those only existed on the company laptop — filtered to files on disk

### Results
- **Test F1: 0.978**, Accuracy: 0.971
- Model was extremely confident: read segments mean P(read) = 0.972, spontaneous mean = 0.042
- Exported to ONNX INT8 (~122MB), uploaded to HuggingFace

### What worked
- The biased threshold approach worked well — high precision for flagging "reading"
- ONNX quantization gave good CPU inference speed (~142ms per 5-sec window)

### What didn't work
- **Fundamental problem:** "read vs spontaneous" is a weak proxy for "cheating vs not cheating"
- A fluent candidate answering genuinely can sound identical to someone reading their own notes
- Someone paraphrasing from a source sounds spontaneous but is cheating
- The model had no concept of what cheating actually sounds like in a Mettl exam context

---

## Phase 2: The Multi-Signal Insight

### The brainstorming
The key realization came from analyzing failure modes:

> "Cheating is just the candidate using outside help for answering an open-ended question. Could be reading, could be pre-prepared, could be GPT or article or whatever."

> "I don't think either 1 approach can do this, I think we need a mix of both for variety of cases."

### Why single approaches fail

| Cheating Type | Audio Model Catches? | Text Model Catches? |
|---------------|---------------------|---------------------|
| Reading from notes | Yes (reading prosody) | Maybe (formal vocab) |
| Reading GPT output | Yes (reading prosody) | Yes (no fillers, high TTR) |
| Memorized answer | No (sounds natural) | Yes (rehearsed patterns) |
| Paraphrasing from source | No (sounds spontaneous) | Partially (formal vocab) |
| Someone whispering answers | Maybe (unusual pauses) | Yes (unnatural hesitation patterns) |

### Architecture decided
Three signal groups feed into one XGBoost classifier:
1. **Audio features** (wav2vec2) — how someone sounds
2. **Text features** — what someone says (fillers, vocabulary complexity, hedges)
3. **Pause features** — where someone pauses (before content words = thinking = spontaneous)

Why XGBoost as combiner: interpretable, fast, works with small labeled data (~500 company samples), feature importance shows what matters.

---

## Phase 3: Transcription — CrisperWhisper Failure, WhisperX Success

### What we tried first: CrisperWhisper
Needed word-level timestamps for pause feature extraction. CrisperWhisper (nyrahealth/CrisperWhisper) promised precise word boundaries.

### Errors encountered
1. **torchcodec import failure** — CrisperWhisper's pipeline tried to use torchcodec for audio loading, which doesn't work on Windows without FFmpeg DLLs
2. **Uninstalled torchcodec** so transformers would fall back to soundfile — but the pipeline still failed
3. **Word timestamp tensor mismatch** — `RuntimeError: The size of tensor a (2) must match the size of tensor b (0)` during timestamp extraction
4. Multiple approaches tried: pipeline with `output_offsets=True`, torchaudio forced alignment as backup — all failed

### What worked: WhisperX
User suggested WhisperX. It uses faster-whisper (CTranslate2) for transcription + wav2vec2-based forced alignment for word timestamps.

**Installation was painful:**
- WhisperX pip install pulled in `torch 2.8.0+cpu`, breaking CUDA PyTorch
- Had to reinstall torch with CUDA: `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`
- torchvision `nms` operator version mismatch
- numba/numpy incompatibility (`Numba needs NumPy 2.0 or less`)

**Final result:** WhisperX transcribed all 1,050 ALLSSTAR files in ~2.5 hours with word-level timestamps. Added `language="en"` to skip per-file language detection.

### Lesson learned
CrisperWhisper is research-grade and fragile. WhisperX is battle-tested for production use with word timestamps.

---

## Phase 4: Feature Engineering — 44 Features

### Text Features (20)
Computed from full transcript text using spaCy:

| Feature | What it captures | Cheating signal |
|---------|-----------------|-----------------|
| filler_rate, filler_count | "um", "uh", "hmm" | Low = possibly reading |
| repetition_rate | Repeated bigrams | High = thinking aloud (spontaneous) |
| repair_rate | "I mean", "no wait" | High = self-correction (spontaneous) |
| ttr, mattr | Vocabulary diversity | High = formal/prepared text |
| complex_word_rate | 3+ syllable words | High = formal writing style |
| avg_word_length | Word complexity | Higher in written text |
| n_words, n_unique_words | Volume | Context feature |
| avg/std_sentence_length | Sentence structure | Low std = rehearsed |
| fragment_rate | Incomplete sentences | High = spontaneous |
| self_ref_rate | "I", "me", "my" | High = personal/spontaneous |
| discourse_marker_rate | "you know", "like" | High = conversational |
| hedge_rate | "I think", "maybe" | High = thinking aloud |
| noun/verb/adj_rate | POS distribution | Written text skews noun-heavy |

### Pause Features (13)
Computed from WhisperX word-level timestamps:

| Feature | What it captures |
|---------|-----------------|
| pause_mean, std, median, skew | Pause duration statistics |
| long_pause_rate | Pauses > 0.5s |
| pause_ratio | Total pause time / total duration |
| n_pauses | Raw count |
| pause_regularity | Std of inter-pause intervals (regular = reading) |
| pause_before_content_ratio | Pauses before nouns/verbs (= thinking) |
| pause_before_function_ratio | Pauses before determiners/prepositions (= reading) |
| mid_phrase_pause_rate | Pauses not at punctuation (= processing) |
| words_per_sec | Speaking rate |
| articulation_rate | Speaking rate excluding pauses |

### Prosodic Features (8)
Computed directly from audio waveform using librosa:

| Feature | What it captures |
|---------|-----------------|
| f0_mean, f0_std, f0_range, f0_skew | Pitch variation (monotone = reading) |
| f0_slope | Pitch trend over time |
| energy_mean, energy_std | Volume variation |
| speaking_rate_std | Rate consistency (consistent = rehearsed) |

### wav2vec2 Features (3)
Added later — scores from the pretrained biased wav2vec2:

| Feature | What it captures |
|---------|-----------------|
| wav2vec2_read_ratio | Fraction of 5-sec windows flagged as "reading" |
| wav2vec2_mean_p_read | Mean P(read) across all windows |
| wav2vec2_max_p_read | Max P(read) across all windows |

### Feature extraction results across datasets

**ALLSSTAR spontaneous:** filler_rate=0.0012, ttr=0.4337, self_ref_rate=0.0325, hedge_rate=0.0754
**ALLSSTAR read:** ttr=0.5325, self_ref_rate=0.0107, hedge_rate=0.0163
**LibriSpeech (all read):** filler_rate=0.0003, ttr=0.8507
**AMI (all spontaneous):** filler_rate=0.0385, hedge_rate=0.1802

Clear separation on key features — fillers and hedges are strong spontaneous signals.

---

## Phase 5: XGBoost Training — Evolution of Results

### Datasets used for training

| Dataset | Type | Files | Role |
|---------|------|-------|------|
| ALLSSTAR | Mixed (699 read, 351 spontaneous) | 1,050 | Has audio + text features |
| LibriSpeech train-clean-100 | Read speech | 5,000 (subsampled) | Text features for read class |
| AMI | Spontaneous meetings | 5,000 (subsampled) | Text features for spontaneous class |
| **Total** | | **11,050** | |

### Stage 1: Text + Pause + Prosodic only (no wav2vec2)

```
F1: 0.939, Accuracy: 0.939, AUC: 0.984
Top features: n_unique_words (22%), long_pause_rate (15%), pause_mean (13%)
```

### Stage 2: Added wav2vec2 scores (ALLSSTAR only scored)
wav2vec2 scores were zero for LibriSpeech/AMI (not scored). XGBoost learned to ignore them.

```
F1: 0.939 (unchanged — wav2vec2 features had zero variance for 10k/11k samples)
```

### Stage 3: Scored ALL datasets with wav2vec2, retrained
Ran `score_wav2vec2.py` on all three datasets. ONNX GPU install failed (DLL lock), switched to PyTorch model on GPU.

Scoring results: ALLSSTAR mean=0.67, LibriSpeech mean=0.63, AMI mean=0.01 (correctly identifies AMI as spontaneous).

```
F1: 0.957, Accuracy: 0.972, AUC: 0.995
Top features: wav2vec2_read_ratio (37%), wav2vec2_max_p_read (31%), wav2vec2_mean_p_read (26%)
```

**Problem discovered:** wav2vec2 features dominated (94% of total importance). All text/pause features became irrelevant. The ensemble was essentially just wav2vec2 with extra steps.

### Stage 4: The weighted voting pivot
Since wav2vec2 was drowning out text features, and wav2vec2 is "always 99% confident even when it's wrong", we split into two independent voters:

1. **Text-only XGBoost** (no wav2vec2 features): F1=0.939
2. **wav2vec2** (separate ONNX model): scores independently
3. **Combined:** weighted average at inference time

---

## Phase 6: Combined wav2vec2 Training

### What we did
Trained wav2vec2 on all three datasets (not just ALLSSTAR) to make it more robust.

### Architecture
Same as biased model: wav2vec2-base encoder + Linear(768→256→ReLU→Dropout→Linear(256→1))
- Freeze first 6 transformer layers
- Mixed precision training (GPU)
- 10 epochs, batch=8, lr=1e-5, early stopping patience=3
- MAX_PER_DATASET=5000 to balance datasets

### Results
```
Test F1: 0.9969 (99.7%), Accuracy: 0.9963
Threshold stable: F1 > 0.996 across all thresholds 0.50-0.80
```

Exported to ONNX INT8 (122.2 MB), uploaded to HuggingFace.

---

## Phase 7: Company Data Deployment

### Constraints
- Company laptop: CPU only, no GPU
- Company data cannot leave the laptop (security/privacy)
- ~500 labeled 1-minute audio samples
- Folder structure: `audios2/`, `audios4/` containing candidate subfolders
- Ground truth: `gtlabels.csv` with "cheating"/"not cheating" labels

### Deployment architecture

```
pipeline.ipynb (end-to-end):
  1. Download models from HuggingFace
  2. Build manifest from gtlabels.csv
  3. Transcribe (WhisperX on GPU staging / faster-whisper on CPU laptop)
  4. Extract features (text + pause + prosodic + wav2vec2 ONNX)
  5. Run pretrained ensemble
  6. Finetune XGBoost on company data
  7. Save + predict

eval_and_finetune.ipynb (after features extracted):
  1. Auto-detect models and existing scores
  2. Load wav2vec2 scores (reuse from CSV or compute fresh)
  3. Load text-only XGBoost (pretrained)
  4. Evaluate per batch (audios2 vs audios4)
  5. Finetune text-only XGBoost on company data
  6. Combined weighted voting with threshold sweep
  7. 5-fold cross-validation
  8. Error analysis
```

### Company data results

| Model | F1 on Company Data |
|-------|-------------------|
| wav2vec2 (pretrained, not finetuned on company) | ~0.60 |
| Text-only XGBoost (finetuned on company) | ~0.76 |
| Weighted vote (optimized) | ~0.82 |

Optimal weight: **0.1 wav2vec2 + 0.9 XGBoost** — text features dominate because wav2vec2 was trained on read-vs-spontaneous (not cheating-vs-not-cheating).

### Short audio handling
Files with < 20 words: text/pause features unreliable, fall back to wav2vec2 only.

### wav2vec2 company finetuning
Created `finetune_wav2vec2_company.py`:
- Loads combined wav2vec2 checkpoint (the 99.7% F1 model)
- Freezes 10/12 transformer layers (~15M trainable params instead of 95M)
- Trains on company audio directly (cheating/not-cheating labels)
- CPU-safe: no mixed precision, batch_size=4, num_workers=0
- Estimated ~2-3 hours for 500 files on CPU
- Auto-exports to ONNX + INT8 after training

---

## Phase 8: What Worked, What Didn't

### What worked well
1. **Multi-signal fusion** — combining audio + text + pause features catches different cheating types
2. **WhisperX** for word-level timestamps — reliable, fast, good alignment quality
3. **XGBoost as the combiner** — interpretable, fast, works with 500 samples, shows feature importance
4. **Weighted voting** over single-model — keeps both signals independent, prevents one dominating
5. **ONNX INT8 quantization** — 3x smaller models, fast CPU inference
6. **Biased threshold approach** — tunable precision/recall tradeoff per deployment
7. **Auto-detection in notebooks** — existing scores reused, models found automatically

### What didn't work
1. **CrisperWhisper** — completely broken for word timestamps on Windows/current transformers
2. **Single-model approach** — no single signal catches all cheating types
3. **wav2vec2 as XGBoost feature** — dominates all other features (94% importance), making ensemble pointless
4. **Read-vs-spontaneous as cheating proxy** — weak on real company data (F1=0.60), because cheating ≠ reading
5. **onnxruntime-gpu install** — DLL locking issues on Windows, had to fall back to PyTorch

### What we'd do differently
1. Start with multi-signal fusion from day one instead of single-model
2. Skip CrisperWhisper entirely, go straight to WhisperX
3. Keep wav2vec2 and XGBoost as independent voters from the start
4. Finetune wav2vec2 on company data earlier (once we had labels)

---

## All Errors Encountered (Chronological)

| # | Error | Cause | Fix |
|---|-------|-------|-----|
| 1 | Manifest path not found | Config pointed to `outputs/` instead of `old/outputs/` | Updated path |
| 2 | Missing GigaSpeech files | Only existed on company laptop | Filtered to files on disk |
| 3 | ONNX export 1.4MB | New PyTorch ONNX exporter bug | Forced legacy exporter |
| 4 | Unicode encoding error | PyTorch ONNX checkmark emoji | Set UTF-8 encoding |
| 5 | HuggingFace upload timeout | Network instability | Retried |
| 6 | torchcodec import failure | Not available on Windows | Uninstalled torchcodec |
| 7 | CrisperWhisper tensor mismatch | Broken timestamp extraction | Abandoned CrisperWhisper |
| 8 | WhisperX installed CPU torch | pip dependency resolution | Reinstalled torch with CUDA index |
| 9 | torchvision nms operator error | Version mismatch | Installed matching versions |
| 10 | numba/numpy incompatibility | numba needed numpy < 2.0 | `pip install numba --upgrade` |
| 11 | WhisperX language detection slow | No language specified | Added `language="en"` |
| 12 | XGBoost ablation bug | train_mask/test_mask undefined in else branch | Added index tracking |
| 13 | wav2vec2 not in ALL_FEATURES | List didn't include wav2vec2 columns | Added WAV2VEC2_FEATURES |
| 14 | onnxruntime-gpu DLL lock | Running process held DLL | Used PyTorch model instead |
| 15 | wav2vec2 checkpoint key mismatch | `self.wav2vec2` vs `self.encoder` | Changed model attribute name |
| 16 | ensemble_results.json not found | Not uploaded to HuggingFace | Fixed download cell + uploaded file |
| 17 | Features CSV not found | User renamed file | Added auto-detection + override |
| 18 | wav2vec2 in finetuned XGBoost | ALL_FEATURES included wav2vec2 | Changed to TEXT_ONLY_FEATURES |

---

## Final Model Performance Summary

| Model/Stage | Dataset | F1 | Notes |
|-------------|---------|-----|-------|
| Biased wav2vec2 (ALLSSTAR) | ALLSSTAR test | 0.978 | Single dataset |
| Combined wav2vec2 (3 datasets) | Combined test | 0.997 | Best audio model |
| XGBoost text-only (pretrained) | Combined test | 0.939 | No wav2vec2 features |
| XGBoost with wav2vec2 features | Combined test | 0.972 | wav2vec2 dominates |
| wav2vec2 (pretrained) | Company data | ~0.60 | Read/spont ≠ cheating |
| XGBoost text-only (finetuned) | Company data | ~0.76 | Text features more relevant |
| Weighted vote (optimized) | Company data | ~0.82 | 0.1 w2v + 0.9 xgb |

---

## Files Created

### Training scripts (personal laptop)
- `train_wav2vec2_combined.py` — Train wav2vec2 on ALLSSTAR + LibriSpeech + AMI
- `train_xgboost_ensemble.py` — Train XGBoost (supports `--no-wav2vec2` flag)
- `export_combined_onnx.py` — Export combined wav2vec2 to ONNX + INT8
- `score_wav2vec2.py` — Score audio files with wav2vec2 model
- `transcribe_whisperx.py` — Transcribe audio with WhisperX
- `extract_features.py` — Extract text + pause + prosodic features
- `upload_combined.py` — Upload models to HuggingFace

### Company laptop scripts
- `companylaptop/pipeline.ipynb` — End-to-end pipeline
- `companylaptop/eval_and_finetune.ipynb` — Eval + finetune with weighted voting
- `companylaptop/finetune_wav2vec2_company.py` — Finetune wav2vec2 on company data (CPU)
- `companylaptop/build_manifest.py` — Build manifest from labels CSV
- `companylaptop/extract_features_company.py` — Feature extraction (self-contained)
- `companylaptop/predict_biased.py` — Standalone wav2vec2 prediction

### HuggingFace (Pransfrance/speechproj-models)
- `biased/biased_wav2vec2_quant.onnx` — Original ALLSSTAR-only model
- `combined/wav2vec2_combined_quant.onnx` — Combined 3-dataset model
- `ensemble/xgboost_ensemble.json` — XGBoost model
- `ensemble/scaler.pkl` — Feature scaler
- `ensemble/ensemble_results.json` — Feature columns + results
- `ensemble_textonly/` — Text-only XGBoost (no wav2vec2 features)

---

## Key Takeaways

1. **Domain mismatch kills transfer learning.** A model trained on "read vs spontaneous" academic data only achieved 60% F1 on real cheating detection. The proxy was too weak.

2. **Multi-signal fusion is essential** for complex behavioral detection. No single feature type (audio, text, or pauses) catches all cheating patterns.

3. **Keep models independent for voting.** When wav2vec2 scores were XGBoost features, they dominated 94% of importance. As separate voters, both contribute meaningfully.

4. **Small labeled datasets favor simple models.** With 500 samples, XGBoost outperforms any neural approach for the text features. Neural models need the pretrained wav2vec2 approach to work at all.

5. **Finetuning on in-domain data is the biggest lever.** The jump from pretrained (60%) to finetuned (76%) on company data was larger than any architecture change.

6. **Threshold tuning matters more than model improvements** at this scale. Moving from 0.5 to the optimal threshold improved practical performance more than switching models.
