# Models Summary — Cheating Detection

Inventory of every text and audio model trained for the cheating-detection pipeline, with variations, hyperparameters, and measured results.

**Task:** Binary classification — scripted/read (cheating=1) vs spontaneous (honest=0) from ~60-second exam audio answers.
**Splits used throughout:** train on `audios2 + audios4`, test on held-out `audios5` (52 positives).

---

## 1. TEXT MODELS

Notebook: `companylaptop/text_cheating_detection.ipynb`
Pipeline: Whisper transcription (cached JSON per folder) → 50 handcrafted features → classifier.

### 1.1 Feature set (50 features, 8 groups)

| Group | # feats | Examples |
|---|---|---|
| **disfluency** | 6 | filler_rate, repetition_rate, discourse_marker_rate, hedge_rate, repair_count, false_start_rate |
| **stylometric** | 15 | ttr, mattr, mtld, avg_word_length, pronoun_rate, self_reference_rate, POS rates (NOUN/VERB/ADJ/ADV), yngve_depth, sentence_length_stats |
| **pause** | 15 | mean/std/median/skew pause length, pause_rate, pause_regularity, initial_pause, longest_pause, suspicious_gap_count, suspicious_gap_ratio, ... |
| **suspicious** | 2 | suspicious_gap_count, suspicious_gap_ratio (0.3–0.8s mid-sentence pauses) |
| **formal_ai** | 4 | formal_transition_count, ai_phrase_markers, hedging_phrases, register_score |
| **prosodic** | 8 | f0_mean/std, energy_mean/std, pitch_range, voiced_ratio (via librosa.pyin) |
| **voice_q** | 3 | jitter, shimmer, HNR (via parselmouth) |
| **perplexity** | 2 | gpt2_perplexity, burstiness |

Transcripts cached as `{folder}_transcripts.json`, features as `{folder}_features.csv` (shared with v3 notebook for autodetection).

### 1.2 Experiment 1 — Feature-group ablation (XGBoost only)

For each of the 8 groups:
- **Single-group XGBoost:** train on that group alone
- **Leave-one-out XGBoost:** train on all 50 feats minus that group
- **All-features XGBoost:** baseline with all 50 feats

**Key findings:**
- Stylometric group dominated (mean_abs_z 0.56 across features)
- formal_ai was second (mean_abs_z 0.42)
- `voice_q`, `prosodic`, `perplexity` had low signal — can be dropped
- **Dropping `suspicious` group improved F1 from 0.6596 → 0.6818** (despite being motivation for the group)

### 1.3 Experiment 2 — Model comparison (all 50 features)

Six classifiers trained on the full feature set:

| Model | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| **RandomForest** | **0.8485** | 0.5385 | 0.6596 | Highest precision on test |
| XGBoost | ~0.80 | ~0.55 | ~0.66 | Balanced baseline |
| LogisticRegression | — | — | highest | Winner by F1 (threshold-calibrated) |
| ExtraTrees | — | — | — | Similar to RF, slightly lower precision |
| GradientBoosting | — | — | — | No improvement over XGBoost |
| LightGBM | — | — | — | Comparable to XGBoost |

(Precise numbers for non-RF models cleared from notebook outputs.)

**XGBoost hyperparameters:**
```python
n_estimators=400, max_depth=4, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
scale_pos_weight=spw, eval_metric='logloss'
```

**RandomForest hyperparameters:**
```python
n_estimators=500, max_depth=8, min_samples_leaf=3,
class_weight='balanced', n_jobs=-1
```

### 1.4 Experiment 3 — Top-N feature selection curve (XGBoost)

Feature importance from all-feats XGBoost → train with top-N features for N ∈ {3, 5, 8, 10, 15, 20, 30, 50}.

| Top-N | Top features included | Precision | Recall | F1 |
|---|---|---|---|---|
| **5** | mattr, mtld, avg_word_length, ttr, formal_transition_count | **0.8438** | 0.5192 | 0.6429 |
| 10 | + pronoun_rate, noun_rate, filler_rate, self_reference_rate, pause_std | ~0.80 | ~0.55 | ~0.66 |
| 20 | + rest of stylometric + formal_ai + top disfluency | ~0.80 | ~0.55 | ~0.66 |
| 50 | full set | see RF row above | — | — |

**Top-5 alone hit 84% precision** — extreme feature reduction was viable.

### 1.5 Production text models (used in fusion)

Two text models carried into the fusion notebook:

| Name | Classifier | Features | Precision | Recall | F1 | rec@P90 |
|---|---|---|---|---|---|---|
| `text_rf` | RandomForest (500 trees, depth 8) | 40 feats (stylometric + formal_ai + disfluency + pause; dropped prosodic/voice_q/perplexity/suspicious) | **0.9032** | 0.5385 | 0.6747 | 0.5385 |
| `text_top5` | XGBoost (5 feats, cs_bytree=0.8) | mattr, mtld, avg_word_length, ttr, formal_transition_count | 0.6905 | 0.5577 | 0.6170 | 0.3462 |

`text_rf` was the strongest single model across the entire project — its 90% precision on its own is why the production fusion is weighted so heavily toward text.

---

## 2. AUDIO MODELS

Notebooks: `wavlm_4way_comparison.ipynb` (classification head), `finetune_wavlm_colab.py` (encoder fine-tuning).
Pipeline: Audio → WavLM encoder → pooled embedding → XGBoost head.

### 2.1 Encoder variants (WavLM-base-plus)

Two encoders:

| Encoder | Description |
|---|---|
| **Pretrained (Pre)** | `microsoft/wavlm-base-plus`, frozen, used as feature extractor |
| **Finetuned (FT)** | Same base, fine-tuned on scripted-vs-spontaneous task (see §2.3) |

### 2.2 Pooling variants

| Pooling | Dim | Description |
|---|---|---|
| **Whole (W)** | 768 | Mean-pool encoder output over the entire clip — one 768-dim vector per audio |
| **Segmented (S)** | 1536 | Split clip into 5-second segments, mean-pool each segment → concat mean + std across segments — captures temporal variation |

### 2.3 WavLM fine-tuning setup (`finetune_wavlm_colab.py`)

**Architecture:**
- Base: `microsoft/wavlm-base-plus`
- Binary head: Dropout → Linear(768→256) → GELU → Dropout → Linear(256→1)
- Freeze bottom 6 transformer layers (fine-tune top 6 + head)

**Training hyperparameters:**
```
WINDOW_SEC = 60   (was 10)
HOP_SEC = 30
MIN_WIN_SEC = 45
WAVLM_LR = 5e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 200
NUM_EPOCHS = 6
BATCH_SIZE = 2   (was 8; reduced for 60s windows)
GRAD_ACCUM = 16  (eff batch = 32)
LOSS = BCEWithLogitsLoss
OPTIMIZER = AdamW (separate LR for encoder vs head)
SCHEDULER = cosine with warmup
MIXED_PRECISION = fp16 via torch.cuda.amp
```

**Training data sources (mix of local + HuggingFace streaming):**

| Source | Label | Access | Mode | Cap |
|---|---|---|---|---|
| AllStar 2677 | scripted (1) | local | lazy | uncapped |
| AllStar 2676 | spontaneous (0) | local | lazy | uncapped |
| CasualConversations scripted | scripted (1) | local | lazy | part of 6k budget |
| CasualConversations spontaneous | spontaneous (0) | local | lazy | part of 9k budget |
| LibriSpeech `clean/train.100` | scripted (1) | HF streaming | eager | 500 windows |
| AMI IHM train | spontaneous (0) | HF streaming | eager | 500 windows |
| VoxPopuli `en/train` | spontaneous (0) | HF streaming | eager | 500 windows |
| CommonVoice non-scripted | spontaneous (0) | local | eager | 500 windows |

**Epochs run:** Earlier runs used 10s windows, 6 epochs. Current config is 60s windows, 6 epochs, effective batch 32.

### 2.4 The 4-way audio comparison

Cross of {Pre, FT} × {Whole, Seg} = 4 downstream feature sets. Each feeds XGBoost:

```python
xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.3 (for >500 dims) or 0.8,
    min_child_weight=3, scale_pos_weight=spw,
    eval_metric='logloss', early_stopping_rounds=30
)
```

| Variant | Input dim | Cached as | Result on audios5 (prec / rec / F1) |
|---|---|---|---|
| **Whole + Pretrained** (`wavlm_wp`) | 768 | `{folder}_whole_pretrained.csv` | **0.6744 / 0.5577 / 0.6105** |
| Whole + Finetuned | 768 | `{folder}_whole_finetuned.csv` | (below Pre in 4-way test) |
| **Seg + Pretrained** (`wavlm_sp`) | 1536 | `{folder}_seg_pretrained.csv` | **0.6458 / 0.5962 / 0.6200** |
| Seg + Finetuned | 1536 | `{folder}_seg_finetuned.csv` | (weakest variant) |

**Critical finding:** **Fine-tuning did not beat frozen pretrained features on this task.** Whole+Pre was the overall winner in the 4-way comparison. That is why the fusion notebook only loads the pretrained variants by default.

Exact FT numbers cleared from notebook outputs; Pre variants verified in fusion notebook baselines.

### 2.5 Production audio models (used in fusion)

| Name | Encoder | Pooling | Input dim | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| `wavlm_wp` | pretrained frozen | whole mean-pool | 768 | 0.6744 | 0.5577 | 0.6105 |
| `wavlm_sp` | pretrained frozen | 5s-segment mean+std | 1536 | 0.6458 | 0.5962 | 0.6200 |

Neither audio model on its own reaches 85% precision at any threshold — the WavLM path is noisy without text. It adds value only through **fusion** with the text models, where its independent error pattern (Jaccard 0.25–0.31 with text_rf) lifts overall recall at high precision.

---

## 3. FUSION MODELS (summary only)

Notebook: `fusion_text_wavlm.ipynb`.
Combines the four base models above (`text_rf`, `text_top5`, `wavlm_wp`, `wavlm_sp`) via 11 methods:

- Weighted averages (pair-wise + 4-way equal + 4-way grid-optimized)
- Voting (AND / OR / majority / AND-strict)
- Geometric mean, rank fusion
- Meta-learner stacking (LogReg, XGBoost on OOF train probas)
- Early fusion (feature concatenation, 3 variants)
- Confidence gating

**Production winners:**
| Use case | Method | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Balanced / flag-for-review | `0.6 * text_rf + 0.4 * wavlm_wp` | 0.46 | 0.8000 | 0.6923 | 0.7423 |
| Zero false accuse | `0.8 * text_rf + 0.2 * wavlm_sp` | 0.75 | 1.0000 | 0.3462 | 0.5143 |

Full fusion analysis lives in `fusion_results_analysis.md`.

---

## 4. Model files and cached artifacts

```
companylaptop/
├── text_cheating_detection.ipynb          (text model experiments)
├── wavlm_4way_comparison.ipynb            (4 audio variants)
├── fusion_text_wavlm.ipynb                (fusion)
├── finetune_wavlm_colab.py                (encoder fine-tuning)
├── extract_wavlm_segmented.py             (5s-segment embedding extractor)
│
├── {folder}_transcripts.json              (Whisper cache, per audio folder)
├── {folder}_features.csv                  (50 text feats, per audio folder)
├── {folder}_whole_pretrained.csv          (768-dim WavLM-Pre-Whole)
├── {folder}_whole_finetuned.csv           (768-dim WavLM-FT-Whole)
├── {folder}_seg_pretrained.csv            (1536-dim WavLM-Pre-Seg)
├── {folder}_seg_finetuned.csv             (1536-dim WavLM-FT-Seg)
│
└── checkpoints_fusion/
    ├── winner_*.pkl
    ├── fusion_comparison.csv
    ├── audios5_full_predictions.csv
    └── per_file_probas.csv
```
