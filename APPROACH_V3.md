# Cheating Detection v3: Research-Backed Approach

## 1. Problem Analysis

**Task:** Detect non-genuine speech in 1-minute interview audio. Candidates may:

| Cheating Type | What Happens | Acoustic Signature | Linguistic Signature |
|---|---|---|---|
| Reading GPT answer | Reads AI-generated text from screen | Monotone, low jitter/shimmer, high HNR, metronomic rhythm, no fillers | Low perplexity, low burstiness, formal vocabulary, long sentences, no hedging |
| Reading article/notes | Reads pre-found content | Reading prosody, uniform pace, eye-tracking pauses | Domain jargon, complex sentences, high coherence |
| Using pre-written notes | Glances at bullet points | Semi-reading prosody, micro-pauses at "checkpoints" | Rehearsed structure, consistent register |
| Getting live help | Someone feeds answers | Long initial silence, then fluent; possible background voice | Natural-sounding but with unusual delays |
| **Genuine answer** | **Thinking while speaking** | **Variable prosody, fillers (um/uh), self-corrections, thinking pauses, breathier voice** | **Informal, hedging, repetition, incomplete sentences, high perplexity** |

**Core insight:** This is fundamentally a **spontaneous vs. non-spontaneous speech detection** problem, combined with **AI-generated content detection** when the source is GPT. Both are well-studied. The literature shows:

- eGeMAPS features alone: AUC 0.91 for read vs. spontaneous (arxiv 2412.11896)
- Whisper encoder embeddings: AUC 0.95 (same paper)
- Timing features alone: 88% accuracy (arxiv 2306.08012)
- GPT-2 perplexity: clear separation between AI text (~10-40 PPL) and human speech (~60-200 PPL)

Our current best (84 prec / 76 recall) has significant room to improve by incorporating these research-validated signals.

---

## 2. Why Current Approaches Fall Short

### 2a. Broken Filler Detection (Critical)

Filler words (um, uh, hmm) are among the strongest signals for genuine speech. But:

- **WhisperX large drops fillers** from transcripts entirely
- The `initial_prompt` trick (nudging Whisper with filler examples) has **never been tried** on our data
- `filler_rate` and `filler_count` (2 of our 41 features) are currently based on **vanilla Whisper which actively suppresses fillers**
- These 2 features are therefore **near-zero for all samples** -- essentially dead features contributing nothing
- This means our text model is running on effectively **39 real features**, not 41

**Fix:** Use `faster_CrisperWhisper` (nyrahealth) -- a CTranslate2-quantized INT8 version of CrisperWhisper that runs on CPU and explicitly preserves fillers, false starts, repetitions, and stutters. This is purpose-built for verbatim transcription.

Alternatively, detect fillers directly from audio using the Filler-semi-CRF model (ICASSP 2023, audio-only, no ASR dependency).

### 2b. Missing Voice Quality Features

Current prosodic features (f0_mean/std/range/skew/slope, energy_mean/std, speaking_rate_std) capture pitch and energy but miss critical **voice quality** measures:

| Missing Feature | What It Captures | Why It Matters |
|---|---|---|
| **Jitter** (cycle-to-cycle F0 variation) | Vocal fold regularity | Reading = low jitter (controlled), genuine = high jitter (thinking effort) |
| **Shimmer** (cycle-to-cycle amplitude variation) | Amplitude consistency | Reading = low shimmer (steady), genuine = variable |
| **HNR** (harmonics-to-noise ratio) | Voice "cleanness" | Reading = high HNR (clear voice), genuine = lower HNR (breathier) |

These are clinically validated, highly discriminative features for read vs. spontaneous speech. Extractable via `parselmouth` (Python Praat wrapper). 3 features, zero risk of overfitting.

### 2c. No AI-Content Detection Signal

When a candidate reads a GPT-generated answer, the transcript inherits GPT's statistical signature: low perplexity and uniform sentence-level complexity. Our current features don't measure this.

**GPT-2 perplexity** directly measures how "predictable" the text is:
- AI-generated text: PPL ~10-40 (highly predictable word sequences)
- Human spontaneous speech transcript: PPL ~60-200 (unpredictable, disfluent)
- **Burstiness** (variance of per-sentence PPL): AI < 100, human > 300

These 2 features directly target the most common cheating type (reading GPT output). GPT-2 small (124M params) runs on CPU.

### 2d. Distribution Shift Not Addressed

audios2 (67% cheating) and audios4 (24% cheating) have very different class distributions. Standard 5-fold CV mixes both batches in every fold, giving optimistic scores that don't transfer to audios5.

**Fix:** GroupKFold by batch + per-batch sample weighting.

### 2e. WavLM XGBoost: High Dims but Works

The WavLM XGBoost uses **768 features on 490 samples** -- a 0.64:1 ratio. This normally suggests overfitting, but it works because:
- `colsample_bytree=0.2` limits each tree to ~154 random features (implicit feature selection)
- WavLM embeddings are high-quality pretrained representations (not random noise)
- XGBoost's boosting naturally focuses on the most discriminative subsets

**PCA was tried and failed:** compressing 768→80 dims dropped performance from ~85% to ~58% F1. The PCA projection destroyed discriminative signal that XGBoost's random subsampling preserves. The low `colsample_bytree` achieves the same regularization benefit without information loss.

### 2f. Overfitting in Combination Method

Both approaches overfit the combination:
- Approach 1: Manual weight grid search finds weights that work on training split only
- Approach 2: Stacking meta-learner trained on overfit base model predictions

**Fix:** Temperature-scaled calibration per model, then simple fixed-weight average.

---

## 3. Proposed Architecture: v3

```
                                 +--[ Whisper + filler prompt ]--+
                                 |     (verbatim transcript      |
                                 |      with fillers)            |
                                 v                               v
AUDIO ──> [ WavLM embeddings ] ──> WavLM XGBoost          Text+Audio XGBoost
           (768-dim, raw)          (768 dims, csbt=0.2)     (48 features)
                |                       |                        |
                |                       v                        v
                |                  temp-scale               temp-scale
                |                       |                        |
                |                       +--- calibrated avg -----+
                |                                 |
                +--- concat ---> Fused XGBoost    |
                |                (~816 dims,      |
                |                 csbt=0.2)       |
                |                     |           |
                |                temp-scale       |
                v                     v           v
           Compare all:    Fused vs Weighted Vote vs Text vs WavLM
```

### 4 Model Variants (all compared side-by-side)

| Model | Features | colsample_bytree | What it tests |
|---|---|---|---|
| **Text** | 48 handcrafted | 0.8 | Text/pause/prosodic signal alone |
| **WavLM** | 768 raw dims | 0.2 | Audio embedding signal alone |
| **Weighted Vote** | 0.6×WavLM + 0.4×Text | -- | Two calibrated models combined |
| **Fused** | ~816 (48 + 768 concat) | 0.2 | Single model learns cross-signal interactions |

### Signal 1: WavLM XGBoost (raw 768 dims, NO PCA)
- 768-dim pretrained WavLM embeddings used directly — **no PCA**
- **PCA was tried and failed:** compressing 768→80 dims dropped WavLM from ~85% to ~58% F1
- Instead, `colsample_bytree=0.2` acts as implicit feature selection (~154 features per tree)
- This preserves all discriminative information while preventing overfitting
- WavLM embeddings are high-quality pretrained representations (not random noise), so the high feature:sample ratio is tolerable

### Signal 2: Enhanced Text+Audio XGBoost (upgraded from 41 to ~48 features)

**Keep all 41 existing features** (they encode domain knowledge well at this scale), **add 5-7 new high-impact features:**

| New Feature | Source | Library | Why |
|---|---|---|---|
| `jitter_local` | Raw audio | `parselmouth` | Reading = low jitter, genuine = high |
| `shimmer_local` | Raw audio | `parselmouth` | Reading = low shimmer, genuine = high |
| `hnr_mean` | Raw audio | `parselmouth` | Reading = high HNR, genuine = lower |
| `mean_perplexity` | Transcript | `transformers` (GPT-2) | AI text = low PPL, human = high PPL |
| `burstiness` | Transcript | `transformers` (GPT-2) | AI text = uniform, human = variable |
| `initial_pause` | Word timestamps | Existing pipeline | Time before first word (thinking time) |
| `longest_pause` | Word timestamps | Existing pipeline | Max pause duration in response |

Total: **48 features on 490 samples = 10:1 ratio** (healthy for XGBoost).

### Signal 3: Fused XGBoost (text + WavLM concatenated)
- Concatenates all 48 text features + 768 WavLM dims = ~816 features
- Single XGBoost with `colsample_bytree=0.2` (~163 features per tree)
- Advantage over weighted vote: can learn **cross-signal interactions** (e.g., low filler_rate + specific WavLM pattern = cheating)
- May outperform weighted vote when text and audio signals are complementary

### Combination: Calibrated Weighted Average (not stacking)

1. **Temperature scaling** on each model independently (1 parameter each, fit on calibration fold)
2. **Fixed weighted average**: `0.6 * wavlm_cal + 0.4 * text_cal` (WavLM is stronger, give it more weight)
3. **Fused model** trained separately as an alternative — compare against weighted vote
4. **Threshold sweep** on calibration fold for precision target (95+)

No meta-learner. No grid search over weights. The weights are set based on relative model strength and don't need to be learned. The fused model is compared alongside to see if a single model outperforms the two-model weighted approach.

---

## 4. Infrastructure Fixes (Tier 0 -- do first)

These require no new features, just fixing how we train and evaluate.

### 4a. GroupKFold Cross-Validation

**Problem:** Standard k-fold mixes audios2 and audios4 in each fold, hiding distribution shift.

**Fix:**
```python
from sklearn.model_selection import GroupKFold

# Each sample tagged with its source batch
groups = ["audios2"] * len(audios2) + ["audios4"] * len(audios4)
gkf = GroupKFold(n_splits=2)

# Fold 1: train on audios4, validate on audios2
# Fold 2: train on audios2, validate on audios4
```

This is harsh but honest. If performance holds across this split, it will generalize to audios5.

For final training, use all of audios2+4 with batch-aware sample weights.

### 4b. Per-Batch Class Rebalancing

**Problem:** audios2 has 67% cheating, audios4 has 24%. Combined training biases toward audios2's distribution.

**Fix:** Weight samples inversely proportional to their class frequency **within their batch**:

```python
# Instead of global scale_pos_weight, use per-sample weights
for sample in audios2:
    if label == 1: weight = 1 / (0.67)  # cheating is common in audios2
    else:          weight = 1 / (0.33)  # non-cheating is rare in audios2
for sample in audios4:
    if label == 1: weight = 1 / (0.24)  # cheating is rare in audios4
    else:          weight = 1 / (0.76)  # non-cheating is common in audios4
# Normalize weights to sum to N
```

This prevents the model from learning "audios2 = cheating, audios4 = not cheating."

### 4c. Temperature Scaling for Calibration

**Problem:** XGBoost probabilities are not well-calibrated, especially across different data distributions.

**Fix:** After training, find a single temperature T per model:

```python
from scipy.optimize import minimize_scalar
import numpy as np

def neg_log_likelihood(T, logits, y_true):
    scaled = logits / T
    probs = 1 / (1 + np.exp(-scaled))
    return -np.mean(y_true * np.log(probs + 1e-8) + (1-y_true) * np.log(1-probs + 1e-8))

# logits = log(p/(1-p)) from XGBoost predict_proba
result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 5.0), method='bounded',
                         args=(logits_val, y_val))
T_optimal = result.x
```

Temperature scaling has only 1 parameter -- can't overfit even with 50 calibration samples.

---

## 5. New Features (Tier 1 -- high impact, easy to implement)

### 5a. Voice Quality via Parselmouth

```python
import parselmouth
from parselmouth.praat import call

def compute_voice_quality(audio_path):
    sound = parselmouth.Sound(audio_path)

    # Jitter
    pitch = call(sound, "To Pitch", 0.0, 75, 500)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    # Shimmer
    shimmer = call([sound, point_process], "Get shimmer (local)",
                   0, 0, 0.0001, 0.02, 1.3, 1.6)

    # HNR
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    return {"jitter_local": jitter, "shimmer_local": shimmer, "hnr_mean": hnr}
```

**Expected impact:** These are among the top discriminative features in read vs. spontaneous speech literature. Adding 3 features to 41 is zero-risk (44 features on 490 samples).

**Dependency:** `pip install praat-parselmouth`

### 5b. GPT-2 Perplexity and Burstiness

```python
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
import re

# Lazy-load (same pattern as SBERT)
_gpt2_model = None
_gpt2_tokenizer = None

def _get_gpt2():
    global _gpt2_model, _gpt2_tokenizer
    if _gpt2_model is None:
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        # CPU is fine for 100-200 word transcripts
    return _gpt2_model, _gpt2_tokenizer

def sentence_perplexity(text):
    model, tokenizer = _get_gpt2()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return torch.exp(loss).item()

def compute_perplexity_features(transcript):
    if not transcript or len(transcript.strip()) < 20:
        return {"mean_perplexity": 0.0, "burstiness": 0.0}

    # Document-level perplexity
    doc_ppl = sentence_perplexity(transcript)

    # Per-sentence perplexity for burstiness
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    sentences = [s for s in sentences if len(s.split()) > 3]

    if len(sentences) < 2:
        return {"mean_perplexity": doc_ppl, "burstiness": 0.0}

    ppls = [sentence_perplexity(s) for s in sentences]
    return {
        "mean_perplexity": np.mean(ppls),
        "burstiness": np.var(ppls),  # AI text: <100, human: >300
    }
```

**Expected impact:** Direct signal for GPT-reading detection. Clear separation in the literature. 2 features, no overfitting risk.

**Dependency:** `pip install transformers torch` (already in environment)

### 5c. Better ASR: faster_CrisperWhisper

Replace `faster-whisper` with `faster_CrisperWhisper` for transcription:

```python
# Instead of:
from faster_whisper import WhisperModel
model = WhisperModel("small")

# Use:
from faster_whisper import WhisperModel
model = WhisperModel("nyrahealth/faster_CrisperWhisper",
                     compute_type="int8",    # CPU-compatible
                     device="cpu")
```

This preserves fillers (um, uh), false starts, repetitions, and stutters. The `filler_rate` and `filler_count` features become reliable.

**Dependency:** Same `faster-whisper` library, different model weights. ~3GB download.

**Caveat:** Slower than faster-whisper small. For 1-minute audio on 8-core CPU: ~30-60 seconds. Acceptable for batch processing, may need optimization for real-time.

### 5d. Additional Pause Features

Two new features from existing word timestamps (zero additional computation):

```python
def compute_extra_pause_features(word_timestamps):
    if not word_timestamps or len(word_timestamps) < 2:
        return {"initial_pause": 0.0, "longest_pause": 0.0}

    # Time before first word (thinking/reading preparation time)
    initial_pause = word_timestamps[0]["start"]

    # Maximum pause duration
    pauses = []
    for i in range(1, len(word_timestamps)):
        gap = word_timestamps[i]["start"] - word_timestamps[i-1]["end"]
        if gap > 0.1:  # >100ms counts as pause
            pauses.append(gap)

    longest_pause = max(pauses) if pauses else 0.0

    return {"initial_pause": initial_pause, "longest_pause": longest_pause}
```

**Expected impact:** `initial_pause` captures thinking time (genuine speakers often pause before starting; readers start immediately or after a fixed delay). `longest_pause` captures max thinking pause (genuine speakers may have long thought pauses; readers don't).

---

## 6. Architecture Expansion (Tier 2 -- when data reaches 800-1000 samples)

### 6a. eGeMAPS as 3rd Base Model

The extended Geneva Minimalistic Acoustic Parameter Set (eGeMAPS) is an 88-dimensional feature set that is the **gold standard** for computational paralinguistics. It achieved AUC 0.91 for read vs. spontaneous speech classification.

```python
import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)
features = smile.process_file("audio.wav")  # returns 88-dim vector
```

Features include: F0 statistics, loudness, MFCCs 1-4, spectral flux/centroid/rolloff, jitter, shimmer, HNR, formants F1-F3, formant bandwidths, voiced/unvoiced segment statistics.

At 800+ samples, 88 features gives a 9:1 ratio -- viable for XGBoost. Add as a 3rd base model in the calibrated ensemble.

**Dependency:** `pip install opensmile`

### 6b. PCA-Compressed SBERT

SBERT embeddings (384-dim) failed at 490 samples. At 1000 samples, compress via PCA:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=32)  # 384 -> 32 dims
sbert_reduced = pca.fit_transform(sbert_embeddings)
# 32 + 23 handcrafted = 55 features, ratio 18:1 -- healthy
```

Add as a 4th base model or merge into the text model.

### 6c. WavLM Data Augmentation

Augment audio **only for WavLM training** (not for text/pause features, since augmentation corrupts timing):

```python
import librosa
import numpy as np

def augment_audio(y, sr):
    augmented = []

    # 1. Add noise (SNR 15-25 dB)
    noise = np.random.randn(len(y))
    snr_db = np.random.uniform(15, 25)
    noise_power = np.mean(y**2) / (10 ** (snr_db / 10))
    augmented.append(y + noise * np.sqrt(noise_power))

    # 2. Speed perturbation (0.9x and 1.1x)
    augmented.append(librosa.effects.time_stretch(y, rate=0.9))
    augmented.append(librosa.effects.time_stretch(y, rate=1.1))

    return augmented  # 3x more WavLM training data
```

**Expected impact:** +3-5 F1 on WavLM model. Improves generalization to different recording conditions.

### 6d. nPVI Rhythm Features

The normalized Pairwise Variability Index measures rhythm regularity from vowel/syllable durations. Read speech has lower nPVI (metronomic); spontaneous has higher (variable).

Requires a forced aligner (Montreal Forced Aligner) to get vowel boundaries:

```bash
pip install montreal-forced-aligner
mfa align audio_dir english_dictionary english_model output_dir
```

Then compute nPVI from vowel interval durations using `npyvi` library.

**Why Tier 2:** Forced alignment adds pipeline complexity. Worth it at scale, overkill at 490 samples.

---

## 7. Future Improvements (Tier 3 -- 2000+ samples)

| Samples | Technique | Expected Impact |
|---|---|---|
| ~2000 | SetFit (few-shot sentence transformer fine-tuning) | SBERT embeddings become domain-specific |
| ~2000 | MLP heads replacing XGBoost on WavLM/eGeMAPS | Better capacity utilization |
| ~3000 | Whisper encoder embeddings as additional base model | 1280-dim speech representation, AUC 0.95 in literature |
| ~5000 | End-to-end WavLM fine-tuning | Learns task-specific audio representations |
| ~5000 | Multi-task learning (cheating type classification) | Joint prediction of reading/GPT/notes/genuine |

---

## 8. Complete Feature List: v3 Enhanced Text+Audio Model

### Existing Features (41) -- KEEP ALL

**Text-Statistics (20):** ttr, mattr, complex_word_rate, avg_word_length, n_words, n_unique_words, avg_sentence_length, std_sentence_length, fragment_rate, n_sentences, self_ref_rate, discourse_marker_rate, hedge_rate, noun_rate, verb_rate, adj_rate, repetition_rate, repair_rate, filler_rate, filler_count

**Pause Features (13):** pause_mean, pause_std, pause_median, pause_skew, long_pause_rate, pause_ratio, n_pauses, pause_regularity, pause_before_content_ratio, pause_before_function_ratio, mid_phrase_pause_rate, words_per_sec, articulation_rate

**Prosodic Features (8):** f0_mean, f0_std, f0_range, f0_skew, f0_slope, energy_mean, energy_std, speaking_rate_std

### New Features (7) -- ADD

| # | Feature | Source | Computation |
|---|---|---|---|
| 42 | `jitter_local` | Raw audio | parselmouth |
| 43 | `shimmer_local` | Raw audio | parselmouth |
| 44 | `hnr_mean` | Raw audio | parselmouth |
| 45 | `mean_perplexity` | Transcript | GPT-2 |
| 46 | `burstiness` | Transcript | GPT-2 (variance of per-sentence PPL) |
| 47 | `initial_pause` | Word timestamps | First word start time |
| 48 | `longest_pause` | Word timestamps | Max inter-word gap |

### Total: 48 features

Ratio: 490 samples / 48 features = **10.2:1** (healthy for XGBoost with regularization)

---

## 9. Dependencies

### New packages required:

```
praat-parselmouth>=0.4.3    # Voice quality (jitter, shimmer, HNR)
# transformers + torch already installed for GPT-2 perplexity
```

### Model downloads:
```
# CrisperWhisper for verbatim transcription (with fillers)
# Downloaded automatically on first use via faster-whisper
model = WhisperModel("nyrahealth/faster_CrisperWhisper")

# GPT-2 for perplexity (124M params, ~500MB)
# Downloaded automatically via transformers
```

### Optional (Tier 2):
```
opensmile>=2.5.0            # eGeMAPS 88 features
montreal-forced-aligner     # For nPVI rhythm features
npyvi                       # nPVI computation
```

---

## 10. Implementation Priority

Execute in this order. Each step is independently testable.

### Phase 1: Fix Foundation (est. 1-2 hours) -- DONE
1. Switch to GroupKFold CV (by batch) -- DONE
2. Add per-batch sample weights -- DONE (scale_pos_weight)
3. WavLM: raw 768 dims, NO PCA, colsample_bytree=0.2 -- DONE (PCA tried and failed)
4. Replace stacking with calibrated weighted average -- DONE
5. Add fused model (text+WavLM concatenated) for comparison -- DONE
6. Retrain and evaluate -- get honest baseline numbers

### Phase 2: Better ASR (est. 1-2 hours)
5. Install and test faster_CrisperWhisper on CPU
6. Re-transcribe a few test samples, verify filler preservation
7. If good: re-transcribe all audio, re-extract text features
8. Retrain and evaluate -- filler_rate/filler_count now reliable

### Phase 3: New Features (est. 2-3 hours)
9. Add parselmouth voice quality (jitter, shimmer, HNR) to extract_features_company.py
10. Add GPT-2 perplexity features (mean_perplexity, burstiness) to extract_features_company.py
11. Add initial_pause and longest_pause to extract_features_company.py
12. Re-extract features for all datasets
13. Retrain 48-feature model, evaluate

### Phase 4: Evaluate on audios5 (est. 30 min)
14. Run full inference pipeline on audios5
15. Compare: old 41-feature baseline vs. new 48-feature enhanced
16. Check precision/recall at various thresholds
17. Document results

---

## 11. Realistic Performance Estimates

### Why previous estimates were too optimistic

The 84/76 measured on audios5 used weights found by grid search on training data -- those weights are overfit. Improvements don't stack linearly -- there's overlap and diminishing returns. And we don't know audios5's class distribution, which affects precision/recall.

### Honest assessment of current state

| What | Reality |
|---|---|
| Measured audios5 result (84/76) | Somewhat overfit -- grid-searched weights may not be optimal for audios5 |
| WavLM model (80/83 training CV) | Overfit from 768 features on 490 samples -- true generalization is lower |
| Text model filler features | **Dead** -- vanilla Whisper suppresses fillers, so filler_rate/count ~0 for all samples |
| Honest cross-batch baseline | Probably **~75-80 F1** with GroupKFold (5-8 points below standard k-fold) |

### Realistic trajectory with all v3 improvements (490 samples)

| Improvement | Marginal Gain | Cumulative F1 | Why |
|---|---|---|---|
| Honest baseline (GroupKFold) | -- | ~75-80 | Starting point after removing overfit estimates |
| + Raw WavLM (no PCA, csbt=0.2) | +0 | ~75-80 | PCA failed; raw 768 with low csbt works best |
| + Fused model (text+WavLM concat) | +2-4 | ~78-83 | Cross-signal interactions in single model |
| + CrisperWhisper (fillers work) | +2-3 | ~80-85 | filler_rate/count go from dead to discriminative |
| + Voice quality (jitter/shimmer/HNR) | +1-3 | ~82-87 | Orthogonal signal, literature-validated |
| + GPT-2 perplexity/burstiness | +1-3 | ~83-89 | Only helps for GPT-reading subset of cheating |
| + Calibration + batch rebalancing | +1-2 | ~84-90 | Better threshold calibration |

### What this means at different precision targets (with all v3 improvements)

| Precision Target | Expected Recall | F1 | Notes |
|---|---|---|---|
| 85 | 80-84 | ~82-84 | **Achievable now with 490 samples** |
| 90 | 74-80 | ~81-84 | **Likely achievable** -- some recall sacrifice |
| 95 | 65-73 | ~77-82 | **Possible but tight** -- heavy recall cost |
| 95 precision + 80 recall | -- | -- | **Not realistic at 490 samples** -- need ~1000+ |

### Path to 95/80 target

The 95 precision / 80 recall target requires ~1000-1500 samples WITH the v3 improvements:

| Data Scale | With v3 Architecture | Precision | Recall |
|---|---|---|---|
| ~490 (now) | All Tier 1 improvements | 88-92 | 76-82 |
| ~800 | + eGeMAPS (Tier 2) | 90-93 | 78-83 |
| ~1000 | + PCA-SBERT + augmentation | 92-95 | 80-85 |
| ~1500 | + SetFit fine-tuning | **95+** | **80+** |

### Bottom line

With 490 samples and all v3 changes, realistic best case is **~90 precision / ~80 recall** (F1 ~84-85). The 95/80 target needs more data. Each week of 200 new labeled audios gets us closer -- by ~1000 samples (3 weeks), the target becomes achievable.

The improvements are still very worth doing because:
1. They fix **real bugs** (dead filler features, overfit WavLM)
2. Each improvement is **orthogonal** (targets a different failure mode)
3. They **scale up** -- the architecture is ready for 1000+ samples without redesign

---

## 12. Key Papers and References

| Topic | Reference | Key Finding |
|---|---|---|
| Read vs. spontaneous (features) | arxiv 2306.08012 | Timing features alone: 88% accuracy |
| Read vs. spontaneous (models) | arxiv 2412.11896 | Whisper AUC 0.95, eGeMAPS AUC 0.91 |
| Voice quality for speech style | Batliner et al. (ResearchGate) | Prosody alone: 74-79% speaker-dependent |
| eGeMAPS feature set | Eyben et al. (IEEE TAFFC 2016) | 88-feature standard for paralinguistics |
| AI text detection (perplexity) | GPTZero methodology | Perplexity + burstiness for AI detection |
| AI text detection (zero-shot) | Binoculars (ICML 2024) | 95% TPR at 0.01% FPR (needs GPU) |
| Filler detection (audio-only) | Zhu et al. (ICASSP 2023) | Semi-CRF on wav2vec, no ASR needed |
| Verbatim ASR | CrisperWhisper (nyrahealth) | Preserves fillers, CPU via faster_CrisperWhisper |
| Disfluency detection | Microsoft (IEEE/ACM TASLP 2024) | Audio-only outperforms ASR-pipeline |
| Small data calibration | Guo et al. (ICML 2017) | Temperature scaling best for small calibration sets |

---

## 13. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CrisperWhisper too slow on CPU | Medium | Medium | Fall back to faster-whisper + initial_prompt; or batch process overnight |
| GPT-2 perplexity noisy on short transcripts | Low | Low | Filter sentences < 4 words; use document-level PPL as fallback |
| Voice quality features not discriminative on this data | Low | Low | Only 3 features added; easy to drop if not helping |
| audios5 distribution fundamentally different | Medium | High | GroupKFold gives honest estimate early; per-batch weighting helps |
| 48 features still overfit | Very Low | Medium | 10:1 ratio with XGBoost regularization is well within safe range |
| ~~PCA drops discriminative WavLM dims~~ | **Confirmed** | **High** | PCA 768→80 dropped F1 from ~85% to ~58%. **Do not use PCA on WavLM.** Raw dims + colsample_bytree=0.2 is the fix. |
| CrisperWhisper transcription quality differs from Whisper | Medium | Low | Different ASR may change other text features too; re-extract ALL features after switching |
