# Recent Models to Try for Cheating Detection

Curated list of models from the past 2-3 years that could replace or
complement the current stack:

- **Current acoustic:** WavLM-base-plus (768d, mean-pool last + layer 9)
- **Current ASR encoder:** Whisper-medium encoder (1024d, mean-pool)
- **Current text:** 55 handcrafted features (disfluency, stylometric, pause,
  prosodic, voice-quality, perplexity-from-GPT-2)
- **Current models:** XGBoost (Tier A picks) + small MLP

The task is read-aloud cheating vs spontaneous speech detection at per-audio
granularity, two clients with different accents and recording chains
(audios2/4/5 = client A; audios6 = client B with IND + PHP regions).

What helps most is **paralinguistic-strong models** (because read-vs-spontaneous
is fundamentally a paralinguistic distinction) and **multilingual robustness**
(because of the IND/PHP region split).

---

## Top-5 to Run First (highest expected return)

| # | Model | Why | Where |
|---|---|---|---|
| 1 | **emotion2vec+** | Pretrained specifically for paralinguistic / emotion. Read-vs-spontaneous is a paralinguistic distinction; this is the closest off-the-shelf match to your task. 768d mean-pool. | `emotion2vec/emotion2vec_plus_large` on HuggingFace |
| 2 | **WavLM-Large** | Direct scale-up from your current base-plus. Same family, no integration cost, ~2.4 SUPERB points better than base-plus on paralinguistic tasks. 1024d mean-pool. | `microsoft/wavlm-large` |
| 3 | **BGE-M3 sentence embeddings of transcript** | Replaces your 55 handcrafted text features with a 1024d semantic-style vector. Multilingual (100+ languages including Indian and Filipino English). Read-aloud answers sound "written" in semantic style; this captures it directly. | `BAAI/bge-m3` |
| 4 | **Qwen2.5-0.5B perplexity** (replace GPT-2) | Your `feat_mean_perplexity` and `feat_burstiness` use GPT-2 from 2019. A modern 0.5B LM is 5-10× better at separating "written prose" from "natural conversation." Drop-in replacement for the perplexity computation in `full_text_features.py`. | `Qwen/Qwen2.5-0.5B` |
| 5 | **NeMo Canary-1B encoder** | The strongest open ASR encoder of 2024-2025. Beats Whisper-large-v3 on the HuggingFace Open ASR Leaderboard (6.67% avg WER). Encoder is a FastConformer; same mean-pool strategy as Whisper. ~1024d. | NVIDIA NeMo |

Implement these one at a time, concatenate the new embedding into the
existing pipeline as another feature block, and re-run R3 from
`COMMANDSIMP.txt` after each addition. If you can only do one: **emotion2vec+**.

---

## 10 Audio Encoder Models (replace / complement WavLM)

### Tier 1: most likely to help

#### 1. emotion2vec+ (FunAudioLLM, 2024)
- **Why:** Self-supervised pretraining specifically targeting emotion and
  paralinguistic features. Read-aloud and spontaneous speech differ in
  exactly the paralinguistic dimensions this model is trained on.
- **HF:** `emotion2vec/emotion2vec_plus_large` (or `_base`, `_seed`)
- **Dim:** 768 (base), 1024 (large)
- **Use:** Mean-pool the last hidden state. Comes with `funasr` pipeline; can
  also extract via `transformers.AutoModel`.
- **Paper:** arXiv:2312.15185 (Dec 2023, integrated into ModelScope/FunASR by Jan 2024)
- **Caveat:** Originally trained on emotion. Your task (read-vs-spontaneous)
  is paralinguistic but not strictly emotional; transfer may not be perfect.
  Still the closest off-the-shelf match.

#### 2. WavLM-Large (Microsoft, 2022)
- **Why:** Same family as your current model, just bigger (300M params vs
  100M). +2.4 average points on SUPERB across 15 paralinguistic tasks.
- **HF:** `microsoft/wavlm-large`
- **Dim:** 1024
- **Use:** Drop-in replacement in `ec2/extract_embeddings.py`. Same
  hidden-states API. Test layer 9 AND last (you found these are equivalent
  for base-plus; might diverge for large).
- **Caveat:** 3× more memory, 3× slower extraction.

#### 3. HuBERT-Large (Meta, 2021/2022)
- **Why:** Complementary to WavLM. Research shows HuBERT is better on
  speaker-related tasks; WavLM is better on paralinguistic. Both are useful;
  concatenating their mean-pools often beats either alone.
- **HF:** `facebook/hubert-large-ls960-ft` or `facebook/hubert-large-ll60k`
- **Dim:** 1024
- **Use:** Mean-pool last hidden state, concat with WavLM.

#### 4. XEUS (CMU + AISG, 2024)
- **Why:** Multilingual SSL trained on 4057 languages, 1M hours. Specifically
  designed to handle accent and language shift — your IND vs PHP gap is
  exactly this kind of distribution problem.
- **HF:** `espnet/xeus`
- **Dim:** 1024
- **Paper:** arXiv:2407.00837
- **Use:** ESPnet-style extraction. Mean-pool last hidden state.
- **Caveat:** Newer ecosystem, may need ESPnet install.

#### 5. ECAPA-TDNN (speaker embeddings, SpeechBrain, 2020/2021)
- **Why:** Small (~6M params), fast, dedicated speaker encoder. Read-aloud
  vs spontaneous reflects a speaker's prosodic style at the moment of
  speaking; a speaker-style embedding captures this without the noise of
  larger SSL models. Excellent complement for ensembling.
- **HF:** `speechbrain/spkrec-ecapa-voxceleb`
- **Dim:** 192
- **Use:** SpeechBrain `EncoderClassifier.encode_batch()`. Already mean-pooled.

### Tier 2: experimental but plausible

#### 6. SenseVoice-Small (FunAudioLLM, 2024)
- **Why:** Multi-task model trained jointly on ASR + emotion + audio event
  detection. The emotion/event branches capture exactly the paralinguistic
  signal you need.
- **HF:** `FunAudioLLM/SenseVoiceSmall`
- **Dim:** ~512 (varies by hook point)
- **Use:** Extract encoder output before the task heads; mean-pool.
- **Caveat:** Designed as an ASR replacement; embedding extraction requires
  poking at intermediate layers, not as turnkey as WavLM.

#### 7. data2vec 2.0 audio (Meta, 2022/2023)
- **Why:** General-purpose SSL with a different training objective from
  WavLM (latent target prediction). Complementary to HuBERT on
  content-related tasks.
- **HF:** `facebook/data2vec-audio-large` or `facebook/data2vec-audio-large-100h`
- **Dim:** 1024
- **Use:** Same hidden_states API as WavLM/HuBERT.

#### 8. BEATs (Microsoft, 2022/2023)
- **Why:** Best general-audio model on AudioSet. Captures non-speech audio
  events (mic clicks, room noise, breath sounds) that may correlate with
  recording conditions of read-aloud setups (e.g., quieter room, closer mic).
- **HF:** `microsoft/BEATs` (community ports)
- **Dim:** 768 (BEATs Iter3+)
- **Use:** Mean-pool patch tokens.
- **Caveat:** Audio-event focused, not speech-focused. Mostly orthogonal
  signal, which is good for ensembling.

#### 9. MMS (Massively Multilingual Speech, Meta, 2023)
- **Why:** Pre-trained on 1107 languages. Strong baseline for accent /
  language shift. Less specialized than XEUS but more mature.
- **HF:** `facebook/mms-1b` (or 300M variants)
- **Dim:** 1280 (1B variant)
- **Use:** Like wav2vec2; mean-pool hidden states.

#### 10. wav2vec 2.0 BERT (Meta SeamlessM4T encoder, 2023)
- **Why:** wav2vec 2.0 augmented with BERT-style training; the encoder used
  inside SeamlessM4T. Strong multilingual semantics.
- **HF:** `facebook/w2v-bert-2.0`
- **Dim:** 1024
- **Use:** AutoModel + mean-pool. Same API as wav2vec2.

---

## 10 ASR / Whisper Alternatives

The current pipeline uses Whisper-medium encoder mean-pool. Better encoders
give better mean-pools for downstream classification.

### Tier 1: drop-in upgrades

#### 1. Whisper-large-v3 (OpenAI, 2023)
- **Why:** Latest Whisper. 10-20% lower WER than -medium across most
  benchmarks. Same encoder API.
- **HF:** `openai/whisper-large-v3`
- **Dim:** 1280
- **Use:** Encoder forward + mean-pool over time. Exact same code path as
  whisper-medium.
- **Caveat:** 3× the model size of medium.

#### 2. Whisper-large-v3-turbo (OpenAI, 2024)
- **Why:** Distilled Whisper-large-v3 with 4-decoder-layer architecture. 8×
  faster than v3 with similar accuracy. Encoder is unchanged from v3, so
  embedding quality is the same.
- **HF:** `openai/whisper-large-v3-turbo`
- **Dim:** 1280
- **Use:** Same as v3 but faster inference.

#### 3. NeMo Canary-1B-v2 (NVIDIA, 2024-2025)
- **Why:** Tops the HF Open ASR Leaderboard at 6.67% avg WER vs Whisper-v3's
  ~8-9%. FastConformer encoder. Open weights.
- **NeMo:** `nvidia/canary-1b`
- **Dim:** 1024
- **Use:** NeMo `EncDecMultiTaskModel`. Extract encoder output, mean-pool.
- **Caveat:** Requires NeMo install. Different API from HuggingFace.

#### 4. NeMo Parakeet-TDT-0.6B-v3 (NVIDIA, 2024-2025)
- **Why:** Smaller (600M) but extremely fast (~10× real-time). 25-language
  ASR. Good when you need cheap extraction for many augmentations.
- **NeMo:** `nvidia/parakeet-tdt-0.6b-v3`
- **Dim:** 1024
- **Use:** Same NeMo pipeline as Canary.

#### 5. Distil-Whisper-large-v3 (HuggingFace, 2023)
- **Why:** Distilled Whisper-large-v3 (decoder reduced). Encoder unchanged
  → same embedding quality but 5-6× faster.
- **HF:** `distil-whisper/distil-large-v3`
- **Dim:** 1280
- **Use:** Same code path as Whisper.

### Tier 2: explore if Tier 1 plateaus

#### 6. SeamlessM4T-v2 (Meta, 2023)
- **Why:** Massive multilingual model (100+ languages). Encoder is w2v-BERT
  2.0 + adapter layers. Strong for cross-accent transfer.
- **HF:** `facebook/seamless-m4t-v2-large`
- **Dim:** 1024 (encoder output)
- **Use:** Extract speech encoder output, mean-pool.

#### 7. Canary-Qwen-2.5B (NVIDIA, June 2025)
- **Why:** Speech-Augmented LM. FastConformer encoder + Qwen3-1.7B
  decoder. The encoder is the same as Canary-1B but trained jointly with
  an LLM, may yield better high-level features.
- **NeMo:** `nvidia/canary-qwen-2.5b`
- **Dim:** 1024
- **Use:** Encoder extraction only — discard the LLM head.

#### 8. SenseVoice-Large (FunAudioLLM, 2024)
- **Why:** Same family as SenseVoice-Small but larger. Multi-task: ASR +
  emotion + event + language ID. The encoder mean-pool carries all those
  signals.
- **HF:** `FunAudioLLM/SenseVoice` (large variant)
- **Dim:** ~1024
- **Use:** Encoder hook + mean-pool.

#### 9. Moonshine (Useful Sensors, 2024)
- **Why:** Tiny (27M / 61M) Whisper-replacement built for real-time edge.
  Compact embeddings (384/512d). Good for distilled student in an
  ensemble.
- **HF:** `UsefulSensors/moonshine`
- **Dim:** 384 (base), 512 (medium)
- **Use:** Encoder mean-pool. Fast extraction.

#### 10. Qwen2-Audio encoder (Alibaba, 2024)
- **Why:** Whisper-large-v3 backbone fine-tuned for audio-language tasks.
  The encoder embeddings reflect richer audio semantics than vanilla
  Whisper.
- **HF:** `Qwen/Qwen2-Audio-7B-Instruct`
- **Dim:** 1280 (Whisper-v3 encoder dims)
- **Use:** Extract audio encoder output before LLM projection.
- **Caveat:** Large model to load just for embeddings.

---

## Text Models — Should You Use One?

**Short answer: yes, and it should help substantially.**

Your current "text features" are 55 hand-crafted statistics (filler rate,
TTR, average sentence length, etc.). These are SHALLOW: they capture
surface form but not meaning or style. A modern sentence-embedding model
projects the full transcript to a vector that encodes semantic style —
e.g. "this sounds like written prose vs spoken conversation."

Why this matters for read-vs-spontaneous specifically:
- Read-aloud answers tend to be **semantically formal**: structured arguments,
  third-person constructions, fewer self-references, fewer hedges.
- Spontaneous answers are **conversational**: false starts, contractions,
  personal anecdotes, mid-sentence revisions.
- A sentence-embedding model trained on diverse text knows the difference
  between these registers natively. Your 55 features approximate it
  shallowly (`self_ref_rate`, `formal_transition_count`, etc.) but a
  semantic vector captures the full distribution.

**Recommended approach:** keep the 55 handcrafted features (cheap, fast,
robust) AND add a 1024d semantic embedding (rich, semantic). Concatenate
into the same input vector your MLP/XGB already consumes.

### 5 Sentence-Embedding Models (semantic style of transcript)

#### 1. BGE-M3 (BAAI, 2024)
- **Why:** Top open-source multilingual embedding (100+ languages, including
  Indian and Filipino English variants). 1024d. Solid MTEB leaderboard
  position. Free, self-hosted, no PII concerns.
- **HF:** `BAAI/bge-m3`
- **Use:** Mean-pool over tokens of the full transcript. `sentence-transformers` library handles this.

#### 2. Qwen3-Embedding-0.6B / 4B / 8B (Alibaba, 2025)
- **Why:** Top of recent MTEB leaderboards. Multiple sizes for cost trade-off.
  0.6B variant is cheap; 8B is best quality. Open weights.
- **HF:** `Qwen/Qwen3-Embedding-0.6B` (also 4B, 8B)
- **Dim:** 1024 (0.6B), 2560 (4B), 4096 (8B)
- **Use:** Sentence-transformers compatible.

#### 3. Gemini Embedding (Google, 2025)
- **Why:** Currently #1 on MTEB (67.71 retrieval score). API-only.
  Multimodal (could embed audio too, in theory, but for now use it for
  transcripts).
- **API:** Google AI Studio
- **Caveat:** PII concern — your transcripts contain candidate data.
  Mercer Mettl's policy probably forbids sending transcripts to a cloud
  embedding API. Only use this if explicitly allowed.

#### 4. NV-Embed-v2 (NVIDIA, 2024)
- **Why:** Tops MTEB for an open model in late 2024. Solid for production
  use. Self-hosted.
- **HF:** `nvidia/NV-Embed-v2`
- **Dim:** 4096
- **Caveat:** Heavy (7B params). Cheaper for batch processing offline.

#### 5. mxbai-embed-large-v1 (Mixedbread, 2024)
- **Why:** Solid English baseline, smaller than NV-Embed (335M params),
  competitive scores. Cheap to host.
- **HF:** `mixedbread-ai/mxbai-embed-large-v1`
- **Dim:** 1024

### 5 LMs for Perplexity (replace GPT-2 in `full_text_features.py`)

Your current `feat_mean_perplexity` and `feat_burstiness` use GPT-2-small,
which is a 2019 model. Modern 0.5B-3B LMs give MUCH sharper perplexity
gradients between conversational and written text. Drop-in replacement —
same code path, just swap the `transformers.AutoModelForCausalLM` model
name.

#### 1. Qwen2.5-0.5B / 1.5B (Alibaba, 2024)
- **Why:** Best perplexity-per-FLOP at small sizes. Strong on
  conversational vs formal distinction.
- **HF:** `Qwen/Qwen2.5-0.5B` or `Qwen/Qwen2.5-1.5B`

#### 2. Llama-3.2-1B / 3B (Meta, 2024)
- **Why:** Industry-standard, broad coverage, well-tested.
- **HF:** `meta-llama/Llama-3.2-1B` (license-gated, request access)

#### 3. Phi-3.5-mini (Microsoft, 2024)
- **Why:** Strong small LM, no license gate.
- **HF:** `microsoft/Phi-3.5-mini-instruct`
- **Caveat:** Trained more on textbook-style data; might score formal text
  too low (the opposite of what you want — formal reads as "natural" to it).
  Worth testing but pick base, not instruct.

#### 4. Gemma-2-2B (Google, 2024)
- **Why:** Solid baseline, no license gate.
- **HF:** `google/gemma-2-2b`

#### 5. SmolLM2-1.7B (HuggingFace, 2024)
- **Why:** Tiny + open. Trained on diverse web data; good at distinguishing
  natural conversation from formal text.
- **HF:** `HuggingFaceTB/SmolLM2-1.7B`

---

## How to Add Any of These to the Existing Pipeline

The pipeline expects `feat_*` columns in `gt.csv` plus WavLM+Whisper
mean-pool arrays in `embeddings_cache.npz`. Adding a new embedding model
is mechanical:

1. **Extract** the new embedding with a small script (parallel to
   `ec2/extract_embeddings.py`). For each `npy_filename`, save a 1024d
   vector keyed by aug name into the same npz format.
2. **Concatenate** in `xgboost_train.py` (`LayerMatrices.__init__`) or
   `neural_baseline_train.py` (`concat()` helper): add the new block to
   `parts = [wavlm, whisper, new_model, feat]`.
3. **Re-run** R3 from `COMMANDSIMP.txt` with the new feature block
   concatenated. The per-client standardization and few-shot pipeline
   handle the bigger input vector unchanged.

For text models the integration is even simpler — compute a sentence
embedding once per transcript, store as 1024d columns in `gt.csv`
(`feat_sbert_0`, `feat_sbert_1`, ...) or as a separate npz. The existing
PCA path will compress them appropriately.

---

## What I'd Actually Do This Week

If I had a week and wanted to push past your current 0.70 a6 F1:

**Day 1:** Add WavLM-Large embeddings to the cache. Re-run R3. Should be
+1 to +2 F1.

**Day 2:** Add emotion2vec+ embeddings. Concatenate. Re-run R3. This is
the highest-expected-return single change.

**Day 3:** Replace the 55 handcrafted text features with BGE-M3 sentence
embeddings. Same recipe.

**Day 4:** Combine all three new embeddings + per-client standardize +
few-shot 0.20 (R3 with everything). This is your new ceiling.

**Day 5:** Swap GPT-2 for Qwen2.5-0.5B in the perplexity computation. If
it helps, keep it; otherwise drop.

**Day 6-7:** Run a single ablation to attribute each improvement (drop
one at a time, re-train) so you can defend the choice when presenting.

If you only have one day: **just emotion2vec+**. The paralinguistic
match to your task is the closest you'll find off-the-shelf.

---

## Sources

- [emotion2vec arXiv:2312.15185](https://arxiv.org/html/2312.15185v1)
- [emotion2vec GitHub](https://github.com/ddlBoJack/emotion2vec)
- [WavLM-Large MTEB / SUPERB comparison](https://www.emergentmind.com/topics/wavlm-model)
- [XEUS arXiv:2407.00837](https://arxiv.org/html/2407.00837v2)
- [XEUS website](https://wanchichen.github.io/pdf/xeus.pdf)
- [NeMo Canary blog](https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/)
- [Canary-1B-v2 & Parakeet-TDT paper](https://arxiv.org/pdf/2509.14128)
- [Open ASR Leaderboard / Whisper alternatives 2025](https://modal.com/blog/open-source-stt)
- [Gladia open-source STT 2026](https://www.gladia.io/blog/best-open-source-speech-to-text-models)
- [MTEB top embedding models](https://modal.com/blog/mteb-leaderboard-article)
- [Gemini Embedding tops MTEB](https://venturebeat.com/ai/new-embedding-model-leaderboard-shakeup-google-takes-1-while-alibabas-open-source-alternative-closes-gap)
- [Qwen3-Embedding family](https://huggingface.co/Qwen)
- [BGE-M3 multilingual embedding](https://huggingface.co/BAAI/bge-m3)
- [Qwen2-Audio Technical Report](https://arxiv.org/html/2407.10759v1)
- [Recent Advances in Speech Language Models survey 2025](https://aclanthology.org/2025.acl-long.682.pdf)
