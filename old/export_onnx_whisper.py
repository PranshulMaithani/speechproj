"""
Export Whisper Medium Classifier to ONNX + INT8 Quantization
=============================================================

Usage:
    python export_whisper_onnx.py --config configs/config.yaml
    python export_whisper_onnx.py --checkpoint checkpoints/whisper_medium_best.pt
"""

import argparse
import time
from pathlib import Path

import yaml
import torch
import numpy as np


def export_whisper(cfg: dict, checkpoint_path: str = None):
    from src.models.train_whisper_cls import WhisperClassifier

    data_root = Path(cfg["paths"]["data_root"])
    ckpt_dir  = data_root / cfg["paths"]["checkpoints_dir"]
    w_cfg     = cfg["training"]["whisper"]

    # Resolve checkpoint
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
    else:
        ckpt_path = ckpt_dir / "whisper_medium_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Checkpoint:  {ckpt_path}")
    print(f"Size:        {ckpt_path.stat().st_size / 1e6:.1f} MB")

    device = torch.device("cpu")   # always export on CPU for portability

    # Load model
    model = WhisperClassifier(
        model_name=w_cfg["model_name"],
        hidden_size=w_cfg.get("hidden_size", 512),
        num_labels=2,
        dropout=0.0,        # no dropout at inference
        freeze_layers=0,    # doesn't matter for export
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_f1 = ckpt.get("val_f1", "?")
    epoch  = ckpt.get("epoch", "?")
    print(f"Loaded:      epoch={epoch}  val_f1={val_f1}")

    # Whisper input: (batch, 80 mel bins, time_frames)
    # time_frames = window_sec * 100  (Whisper processes at 100 frames/sec)
    window_sec  = cfg["audio"]["window_sec"]
    sr          = cfg["audio"]["sample_rate"]
    n_frames    = 3000   # 500 frames for 5sec
    dummy_input = torch.zeros(1, 80, n_frames)

    print(f"\nWindow:      {window_sec}s  →  input shape: (1, 80, {n_frames})")

    # ── Step 1: Export fp32 ONNX ──────────────────────────────
    onnx_path = ckpt_dir / "speech_classifier_whisper_medium.onnx"
    print(f"\nExporting fp32 ONNX: {onnx_path.name}")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=14,
        input_names=["input_features"],
        output_names=["logits"],
        dynamic_axes={
            "input_features": {0: "batch_size"},
            "logits":         {0: "batch_size"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"Validated OK  |  Size: {onnx_path.stat().st_size / 1e6:.1f} MB")

    # ── Step 2: INT8 quantization ─────────────────────────────
    quant_path = ckpt_dir / "speech_classifier_whisper_medium_quant.onnx"
    print(f"\nQuantizing to INT8: {quant_path.name}")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(onnx_path),
            str(quant_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        print(f"Quantized size: {quant_path.stat().st_size / 1e6:.1f} MB")

        # Verify quantized model runs
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(quant_path), providers=["CPUExecutionProvider"]
        )
        test_out = sess.run(
            ["logits"],
            {"input_features": dummy_input.numpy()},
        )
        print(f"Inference test OK  |  output shape: {test_out[0].shape}")

    except ImportError:
        print("onnxruntime quantization not available — skipping INT8 step")
        quant_path = onnx_path

    # ── Step 3: Benchmark on CPU ──────────────────────────────
    print(f"\nBenchmarking CPU inference...")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(quant_path), providers=["CPUExecutionProvider"]
        )
        test_batch = np.random.randn(4, 80, n_frames).astype(np.float32)

        # Warmup
        for _ in range(3):
            sess.run(["logits"], {"input_features": test_batch})

        t0     = time.perf_counter()
        n_runs = 20
        for _ in range(n_runs):
            sess.run(["logits"], {"input_features": test_batch})
        elapsed = (time.perf_counter() - t0) / n_runs

        ms_per_batch  = elapsed * 1000
        ms_per_window = ms_per_batch / 4
        sec_per_file  = ms_per_window * 23 / 1000   # ~23 windows per 60s file

        print(f"  Batch-4:         {ms_per_batch:.1f}ms")
        print(f"  Per window:      {ms_per_window:.1f}ms")
        print(f"  Per 60s file:    {sec_per_file:.1f}s")
        print(f"  219 files:       {219 * sec_per_file / 60:.1f} min")

    except Exception as e:
        print(f"Benchmark skipped: {e}")

    print(f"\n{'='*60}")
    print(f"Export complete")
    print(f"  fp32:  {onnx_path.name}  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    print(f"  int8:  {quant_path.name}  ({quant_path.stat().st_size/1e6:.1f} MB)  ← use this")
    print(f"\nCopy to laptop:")
    print(f"  {quant_path}")
    print(f"\nRun inference:")
    print(f"  python predict_cpu.py --audio audios2/ \\")
    print(f"    --model checkpoints/{quant_path.name} \\")
    print(f"    --model-type whisper \\")
    print(f"    --window-sec {window_sec} \\")
    print(f"    --output outputs/results_whisper_medium.json \\")
    print(f"    --no-silero")

    return quant_path


def main():
    parser = argparse.ArgumentParser(
        description="Export Whisper Medium classifier to ONNX"
    )
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint (default: checkpoints/whisper_medium_best.pt)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    export_whisper(cfg, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()