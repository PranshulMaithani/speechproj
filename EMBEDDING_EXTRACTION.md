# How the Acoustic Embeddings Are Extracted (WavLM + Whisper)

*From raw audio to a fixed-size vector — the processing steps for each model, and exactly
how the code does it.*

Both encoders are run **once** per audio (× per augmentation) by `ec2/extract_embeddings.py`
into a stamped `embeddings_cache.npz`; downstream training reads the cache instead of
re-running the models. Everything here is **inference-only** (`@torch.no_grad()`), the
weights are **frozen** (never fine-tuned), and both models end the same way for us:
**take a layer's frame sequence → mean-pool over time → one vector.**

Shared constants (`extract_embeddings.py`):

| Constant | Value | Meaning |
|---|---|---|
| `TARGET_SR` | 16000 | all audio is 16 kHz mono |
| `WAVLM_ID_DEFAULT` | `microsoft/wavlm-base-plus` | 768-d, 12 layers |
| `WHISPER_ID_DEFAULT` | `openai/whisper-medium` | 1024-d encoder, 24 layers |
| `WAVLM_CHUNK_SEC` | 20.0 | WavLM processing window |
| `WHISPER_CHUNK_SEC` | 30.0 | Whisper's native window |

---

## 1. WavLM — raw waveform → 768-d

**Input:** an anonymized **16 kHz mono waveform** (a 1-D array of samples). WavLM eats the
raw waveform directly — no spectrogram.

1. **Chunking.** Split into **20 s** windows (`WAVLM_CHUNK_SEC`); skip any chunk < 0.4 s.
2. **CNN feature encoder (waveform → frames).** 7 temporal conv layers downsample
   16,000 samples/s to **~50 feature vectors/s** — one vector per **~20 ms frame**.
3. **Project + position.** Frames are layer-normalized, projected to width **768**, and given
   convolution-based positional information.
4. **12 Transformer encoder layers.** Self-attention (with WavLM's gated relative position
   bias) makes each frame context-aware. All layer outputs are exposed
   (`output_hidden_states=True`): `hidden_states[0…12]`, each `[frames × 768]`.
5. **Pick layers.** We take **`last`** (`= hidden_states[12]`) and **layer 9**
   (`hidden_states[9]`) — still a *sequence* of 768-d frame vectors.
6. **Mean-pool over time.** Average the frame vectors → **one 768-d vector per chunk** per layer.
7. **Combine chunks — length-weighted.** Average the per-chunk vectors, **weighted by each
   chunk's frame count** (longer chunk counts more) → the final **768-d** vector per layer.
8. **Zero-vector fallback** if no chunk was long enough.

---

## 2. Whisper — raw waveform → 1024-d (encoder only)

**Input:** the same **16 kHz mono waveform**. Whisper first turns it into a spectrogram.

1. **Chunking.** Split into **30 s** windows (`WHISPER_CHUNK_SEC`, Whisper's native window);
   skip any chunk < 0.4 s.
2. **Log-mel feature extraction.** `WhisperFeatureExtractor` converts each chunk into a
   **log-mel spectrogram (80 mel bins)**, padded/truncated to exactly 30 s → a fixed
   **3000-frame** `input_features` tensor.
3. **Conv stem.** 2 convolution layers process the mel and downsample along time → a sequence
   of **1024-d** vectors (~1500 frames for 30 s).
4. **+ Sinusoidal positional encoding** so the transformer knows frame order.
5. **24 Transformer encoder blocks.** Self-attention builds a context-aware representation.
   We call **`model.encoder(...)` only** and take its `last_hidden_state` — the **decoder is
   never used**, so no text is ever decoded.
6. **Mean-pool over time.** Average the encoder's frame vectors → **one 1024-d vector per chunk**.
7. **Combine chunks — simple mean.** Average the per-chunk vectors (unweighted, unlike
   WavLM's length-weighting) → the final **1024-d** vector.
8. **Zero-vector fallback** if no chunk was long enough.

> Detail worth knowing: because the feature extractor pads each chunk to a full 30 s, the
> mean-pool in step 6 averages over the padded region too. It's consistent for every audio,
> so batches stay comparable.

---

## 3. WavLM vs Whisper at a glance

| | **WavLM-base-plus** | **Whisper-medium (encoder)** |
|---|---|---|
| Input to the model | raw **waveform** | **log-mel spectrogram** (80 bins) |
| Frontend | 7 conv layers (→ ~20 ms frames) | 2 conv layers on the mel |
| Transformer | 12 self-attention layers | 24 encoder blocks (decoder unused) |
| Layers we keep | **`last` + layer 9** | last encoder layer only |
| Chunk window | 20 s | 30 s |
| Chunk combine | **length-weighted** mean | simple mean |
| Output dim | **768** | **1024** |
| Common ending | mean-pool over time → one vector | mean-pool over time → one vector |

---

## 4. The code — how we extract them

Both extractors live in `ec2/extract_embeddings.py` and are called once per (file, aug).

### 4a. WavLM — `extract_wavlm_meanpool`

```python
@torch.no_grad()
def extract_wavlm_meanpool(wav, model, device, layers):
    """Return {layer_tag: (768,)} for each requested layer."""
    chunk = int(WAVLM_CHUNK_SEC * TARGET_SR)                       # 20 s window
    chunks = [wav] if len(wav) <= chunk else \
             [wav[i:i + chunk] for i in range(0, len(wav), chunk)]
    pools = {l: [] for l in layers}
    weights = []
    for c in chunks:
        if len(c) < TARGET_SR * 0.4:                              # skip < 0.4 s
            continue
        x = torch.from_numpy(c).float().unsqueeze(0).to(device)   # raw waveform in
        out = model(x, output_hidden_states=True)                 # all layer states
        weights.append(out.last_hidden_state.shape[1])            # #frames in this chunk
        for l in layers:
            layer_out = out.last_hidden_state if l == "last" \
                        else out.hidden_states[int(l)]             # 'last' or hidden_states[9]
            pools[l].append(layer_out.mean(dim=1).squeeze(0)      # mean-pool over time
                            .cpu().numpy())
    if not weights:
        z = np.zeros(model.config.hidden_size, dtype=np.float32)
        return {l: z.copy() for l in layers}
    W = np.asarray(weights, dtype=np.float32)                     # length-weighted
    return {l: (np.stack(pools[l]) * (W[:, None] / W.sum())).sum(0).astype(np.float32)
            for l in layers}
```

### 4b. Whisper — `extract_whisper_meanpool`

```python
@torch.no_grad()
def extract_whisper_meanpool(wav, model, feat, device):
    """Whisper-medium ENCODER mean-pool -> (1024,)."""
    chunk = int(WHISPER_CHUNK_SEC * TARGET_SR)                    # 30 s window
    chunks = [wav] if len(wav) <= chunk else \
             [wav[i:i + chunk] for i in range(0, len(wav), chunk)]
    pooled = []
    for c in chunks:
        if len(c) < TARGET_SR * 0.4:                             # skip < 0.4 s
            continue
        feats = feat(c, sampling_rate=TARGET_SR,                 # -> log-mel (80 bins)
                     return_tensors="pt").input_features.to(device)
        enc = model.encoder(feats).last_hidden_state            # ENCODER ONLY (no decoder)
        pooled.append(enc.mean(dim=1).squeeze(0).cpu().numpy()) # mean-pool over time
    if not pooled:
        return np.zeros(model.config.d_model, dtype=np.float32)
    return np.stack(pooled, axis=0).mean(axis=0).astype(np.float32)   # simple mean of chunks
```

### 4c. Orchestration (the driver loop)

```python
# load frozen models once
wavlm        = WavLMModel.from_pretrained(args.wavlm_id).to(device).eval()
whisper      = WhisperModel.from_pretrained(args.whisper_id).to(device).eval()
whisper_feat = WhisperFeatureExtractor.from_pretrained(args.whisper_id)

for fn, augs_for_file in by_file.items():          # only files/augs missing from cache
    wav = np.load(npy_dir / fn).astype(np.float32) # anonymized 16 kHz waveform
    for a in augs_for_file:
        wav_a = wav if a == "orig" else augs_callable[a](samples=wav, sample_rate=TARGET_SR)
        res = extract_wavlm_meanpool(wav_a, wavlm, device, layers_needed)  # {layer: 768}
        for l, emb in res.items():
            wavlm_data[(l, a)][row_idx] = emb
        whisper_data[a][row_idx] = extract_whisper_meanpool(wav_a, whisper, whisper_feat, device)

save_cache(out_path, filenames, aug_names, layers, wavlm_data, whisper_data,
           args.wavlm_id, args.whisper_id)         # stamps model IDs into the .npz
```

Key properties of the extraction step:

- **Extract-once / incremental.** The driver only processes `(file, aug, layer)` combinations
  that are still missing from the cache; re-running with the same inputs is a no-op.
- **Stamped cache.** `save_cache` writes the WavLM/Whisper model IDs into the `.npz`; the
  loader refuses to mix a base-model cache (768-d) with a large-model cache (1024-d).
- **Deterministic.** `orig` embeddings are deterministic; augmentations are seeded. This is
  what makes the whole downstream 30-variant sweep bit-reproducible.
- **Keys.** Arrays are stored as `wavlm_<layer>_<aug>` (e.g. `wavlm_9_noise`) and
  `whisper_<aug>`, aligned to `filenames`, so `_data_pipeline.load_cache_reindexed` can line
  them up with `gt.csv` rows for any run.

---

### Presentation one-liner
*"Each audio is run once through both frozen encoders — WavLM on the raw waveform, Whisper on
its log-mel spectrogram — and in both cases we take a transformer layer's per-frame output and
mean-pool it over time into a single vector (768 for WavLM, 1024 for Whisper), which we cache
and reuse."*

*Source of truth: `ec2/extract_embeddings.py`. Downstream fusion + model: `METHODOLOGY.md`.*
