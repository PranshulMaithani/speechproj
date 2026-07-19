#!/usr/bin/env python3
"""WAV -> trained models pipeline, GPU transcription variant.

Identical to `wav_pipeline.py` in every stage EXCEPT Stage 1 (transcription),
which here defaults to running **faster-whisper on the GPU** (`device=cuda`,
`compute_type=float16`) instead of CPU int8 -- the slow part of the CPU pipeline.

Why a separate copy (not a flag): the CPU pipeline stays the untouched, known-good
reference; this file is the "faster" one your seniors asked for. Everything else
is reused from `wav_pipeline.py`, so the two can never silently diverge.

Same output as the CPU pipeline:
  * SAME transcription model (default kept -> identical to audios2..6), SAME rich
    word-timestamped format, SAME filler-priming prompt -> the 55 handcrafted
    feat_* stay comparable and disfluencies (um / uh / restarts) are still kept.
    Only the compute backend changes, so it's much faster on a GPU box.
  * SAME anonymisation (CID -> encoded id + local cid_mapping.json), SAME npy,
    SAME embeddings, SAME training.

NOTE on exactness: GPU float16 decoding can differ from CPU int8 by tiny numeric
amounts on a few tokens; the model, prompt, and output schema are identical, so
the features are equivalent (re-validate once if you need bit-identical parity).

Requires a CUDA GPU + faster-whisper's CUDA libraries (cuBLAS/cuDNN). On a box
with no GPU this errors in Stage 1 -- use `wav_pipeline.py` (CPU) there instead.

Run (full, GPU transcription):
    python pipeline/wav_pipeline_gpu.py \
        --batches auto \
        --out_dir data/runs/retrain_from_wav_gpu \
        --train_config ec2/configs/train_job.example.yaml

Override the GPU compute type if needed (e.g. an older card): --compute_type int8_float16
"""
from __future__ import annotations

import sys

# Reuse the entire CPU pipeline; only the transcription-stage defaults change.
from wav_pipeline import build_argparser, run_pipeline


def main() -> int:
    """Same pipeline as wav_pipeline.py, but Stage 1 transcription defaults to the
    GPU (device=cuda, compute_type=float16). All other flags are inherited."""
    ap = build_argparser(default_device="cuda", default_compute_type="float16")
    ap.description = "WAV -> trained models pipeline (GPU transcription)"
    args = ap.parse_args()
    print("[wav_pipeline_gpu] GPU transcription variant "
          "(faster-whisper device=cuda). Same model/prompt/output as the CPU pipeline.")
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
