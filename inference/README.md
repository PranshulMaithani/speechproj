# Inference

Self-contained scoring of raw audio with already-trained models. Full detail +
every flag: [`docs/REF_INFERENCE.md`](../docs/REF_INFERENCE.md).

## Layout
```
inference/
  audios/    <- your audio (any names). Each child dir = an "audio group".
             Recurses; only *_25 / *_26 / *_27 wavs are scored.
  models/    <- one subfolder per model, each with:
             model.pt  scaler.joblib  inference_meta.json  threshold.txt  [pca.joblib]
  data/      <- cache (npy, transcript, features, embeddings) -- auto-created
  results/   <- inference_results.xlsx (one sheet per model) -- auto-created
  run_inference.py
```

## Run
```bash
python inference/run_inference.py                                  # CPU transcription
python inference/run_inference.py --transcribe_device cuda --transcribe_compute_type float16
python inference/run_inference.py --force                          # recompute cache
```

## What it does
`wav -> npy (16 kHz) -> transcript -> 55 features -> WavLM+Whisper embeddings ->
per model: 1847-d vector -> scaler -> pca -> MLP -> sigmoid -> apply threshold.txt`.

`threshold.txt` holds one float (e.g. `0.74`); **probability ≥ threshold → CHEAT (1)**,
else GENUINE (0). Output: `results/inference_results.xlsx` — a `summary` sheet plus one
sheet per model (columns: audio_group, audio_name, question, path, probability, threshold,
result, decision), and per-model CSVs.
