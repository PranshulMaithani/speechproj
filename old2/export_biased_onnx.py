"""
Export Biased Wav2Vec2 (single-neuron) to ONNX + INT8 quantization.

Usage:
    python export_biased_onnx.py

Output:
    checkpoints_biased/biased_wav2vec2.onnx         (fp32)
    checkpoints_biased/biased_wav2vec2_quant.onnx   (int8, use this)
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "old"))

from train_biased_wav2vec2 import BiasedSpeechClassifier, READ_THRESHOLD

CKPT_DIR = Path("checkpoints_biased")
CKPT_PATH = CKPT_DIR / "biased_wav2vec2_best.pt"
SAMPLE_RATE = 16000
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)


def main():
    device = torch.device("cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    w2v_cfg = checkpoint["config"]
    print(f"  Epoch {checkpoint['epoch']}, val_f1={checkpoint['val_f1']:.4f}")
    print(f"  Read threshold: {checkpoint.get('read_threshold', READ_THRESHOLD)}")

    # Build model with no freezing/dropout for inference
    model = BiasedSpeechClassifier(
        model_name=w2v_cfg["model_name"],
        hidden_size=w2v_cfg["hidden_size"],
        dropout=0.0,
        freeze_layers=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Step 1: Export to ONNX fp32
    onnx_path = CKPT_DIR / "biased_wav2vec2.onnx"
    print(f"\nExporting to ONNX fp32: {onnx_path}")

    dummy_input = torch.zeros(1, WINDOW_SAMPLES)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=14,
        input_names=["input_values"],
        output_names=["logit"],
        dynamic_axes={
            "input_values": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX validated OK — {onnx_path.stat().st_size / 1e6:.1f} MB")

    # Step 2: INT8 dynamic quantization
    quant_path = CKPT_DIR / "biased_wav2vec2_quant.onnx"
    print(f"\nQuantizing to INT8: {quant_path}")

    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        str(onnx_path),
        str(quant_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"  Quantized — {quant_path.stat().st_size / 1e6:.1f} MB")

    # Step 3: Verify quantized model
    import onnxruntime as ort

    sess = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])
    test_input = np.random.randn(1, WINDOW_SAMPLES).astype(np.float32)
    output = sess.run(["logit"], {"input_values": test_input})
    print(f"  Inference test OK — output shape: {output[0].shape}")

    # Step 4: Benchmark
    print("\nBenchmarking CPU inference...")
    test_batch = np.random.randn(8, WINDOW_SAMPLES).astype(np.float32)
    # Warmup
    for _ in range(3):
        sess.run(["logit"], {"input_values": test_batch})

    t0 = time.perf_counter()
    n_runs = 20
    for _ in range(n_runs):
        sess.run(["logit"], {"input_values": test_batch})
    elapsed = (time.perf_counter() - t0) / n_runs

    ms_per_batch = elapsed * 1000
    ms_per_window = ms_per_batch / 8
    print(f"  Batch-8: {ms_per_batch:.1f}ms ({ms_per_window:.1f}ms/window)")
    print(f"  ~{ms_per_window * 25 / 1000:.1f}s per 1-min file")

    print(f"\nDone. Use: {quant_path}")


if __name__ == "__main__":
    main()
