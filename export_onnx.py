"""
ONNX Export + INT8 Quantization
==================================
Exports the trained SpeechClassifier to ONNX with INT8 dynamic quantization.
On CPU this is typically 3-5x faster than PyTorch fp32 inference.

Run this ONCE on the training machine (GPU), then copy onnx/ to the laptop.

Usage:
    python export_onnx.py --config configs/config.yaml

Output:
    checkpoints/speech_classifier.onnx           (fp32, ~380MB)
    checkpoints/speech_classifier_quant.onnx     (int8, ~95MB, use this)
"""

import argparse
from pathlib import Path

import yaml
import torch
import numpy as np


def export_onnx(cfg: dict):
    data_root = Path(cfg["paths"]["data_root"])
    ckpt_dir = data_root / cfg["paths"]["checkpoints_dir"]
    w2v_cfg = cfg["training"]["wav2vec2"]

    device = torch.device("cpu")  # Export on CPU for portability

    # Load the trained model
    from src.models.train_wav2vec2 import SpeechClassifier
    model = SpeechClassifier(
        model_name=w2v_cfg["model_name"],
        hidden_size=w2v_cfg["hidden_size"],
        num_labels=2,
        dropout=0.0,
        freeze_layers=0,
    )
    checkpoint = torch.load(
        ckpt_dir / "wav2vec2_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')}, val_f1={checkpoint.get('val_f1', '?'):.4f})")

    # Fixed-length input: 5 seconds at 16kHz = 80000 samples
    sr = cfg["audio"]["sample_rate"]
    window_samples = int(cfg["audio"]["window_sec"] * sr)
    dummy_input = torch.zeros(1, window_samples)

    # ------ Step 1: Export to ONNX (fp32) ------
    onnx_path = ckpt_dir / "speech_classifier.onnx"
    print(f"\nExporting to ONNX fp32: {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=14,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=True,
        dynamo=False,          # Use legacy TorchScript-based exporter (no onnxscript needed)
    )

    # Verify the export
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX model validated OK")
    print(f"  Size: {onnx_path.stat().st_size / 1e6:.1f} MB")

    # ------ Step 2: INT8 dynamic quantization ------
    quant_path = ckpt_dir / "speech_classifier_quant.onnx"
    print(f"\nQuantizing to INT8: {quant_path}")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        # Only quantize MatMul/Gemm (linear layers), NOT Conv layers
        # Conv quantization (ConvInteger) is not supported by all ONNX Runtime builds
        quantize_dynamic(
            str(onnx_path),
            str(quant_path),
            weight_type=QuantType.QInt8,
            nodes_to_exclude=[],
            op_types_to_quantize=["MatMul", "Gemm"],  # Skip Conv
        )
        print(f"  Quantized size: {quant_path.stat().st_size / 1e6:.1f} MB")

        # Quick inference test
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(quant_path),
            providers=["CPUExecutionProvider"],
        )
        test_input = np.random.randn(1, window_samples).astype(np.float32)
        output = sess.run(["logits"], {"input_values": test_input})
        print(f"  Quantized test inference OK — shape: {output[0].shape}")

    except ImportError:
        print("  onnxruntime not installed, skipping quantization.")
        print("  Install with: pip install onnxruntime onnxruntime-tools")
        quant_path = onnx_path  # Fall back to fp32

    # ------ Step 3: Benchmark on CPU ------
    print("\nBenchmarking CPU inference speed...")
    try:
        import onnxruntime as ort
        import time

        sess = ort.InferenceSession(
            str(quant_path),
            providers=["CPUExecutionProvider"],
        )
        test_input = np.random.randn(8, window_samples).astype(np.float32)  # batch=8

        # Warm up
        for _ in range(3):
            sess.run(["logits"], {"input_values": test_input})

        # Time it
        t0 = time.perf_counter()
        N = 20
        for _ in range(N):
            sess.run(["logits"], {"input_values": test_input})
        elapsed = (time.perf_counter() - t0) / N

        ms_per_batch = elapsed * 1000
        ms_per_window = ms_per_batch / 8
        windows_per_min_file = 25
        sec_per_file = ms_per_window * windows_per_min_file / 1000

        print(f"  Batch-8 inference: {ms_per_batch:.1f}ms  ({ms_per_window:.1f}ms / window)")
        print(f"  Estimated time per 1-min file: {sec_per_file:.1f}s")
        print(f"  Estimated time for 475 files:  {475 * sec_per_file / 60:.1f} min")

    except Exception as e:
        print(f"  Benchmark skipped: {e}")

    print(f"\nExport complete!")
    print(f"  Copy to laptop: {quant_path}")
    print(f"  Also copy:      {ckpt_dir / 'xgboost_baseline.json'}")
    print(f"  Also copy:      {ckpt_dir / 'xgboost_scaler.pkl'}")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    export_onnx(cfg)


if __name__ == "__main__":
    main()
