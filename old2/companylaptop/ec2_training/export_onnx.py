"""
Export trained wav2vec2 to ONNX + INT8 quantization.
====================================================
Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint checkpoints/wav2vec2_best.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

WINDOW_SAMPLES = 80000  # 5 sec @ 16kHz


class BiasedSpeechClassifier(nn.Module):
    """Same architecture as train.py — needed for loading weights."""

    def __init__(self, model_name="facebook/wav2vec2-base", hidden_size=256,
                 dropout=0.0, freeze_layers=0):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_values):
        outputs = self.encoder(input_values).last_hidden_state
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="Export wav2vec2 to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wav2vec2_best.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  Epoch: {ck.get('epoch', '?')}, Val F1: {ck.get('val_f1', '?')}")

    # Build model (no freezing/dropout for inference)
    model = BiasedSpeechClassifier(dropout=0.0, freeze_layers=0)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # Export FP32 ONNX
    onnx_path = output_dir / "wav2vec2_trained.onnx"
    dummy = torch.zeros(1, WINDOW_SAMPLES)

    print(f"\nExporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model, dummy, str(onnx_path),
        opset_version=14,
        input_names=["input_values"],
        output_names=["logit"],
        dynamic_axes={
            "input_values": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    # Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  FP32: {size_mb:.1f} MB")

    # Quantize to INT8
    quant_path = output_dir / "wav2vec2_trained_quant.onnx"
    print(f"\nQuantizing to INT8: {quant_path}")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(onnx_path), str(quant_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    quant_mb = quant_path.stat().st_size / 1e6
    print(f"  INT8: {quant_mb:.1f} MB ({100*quant_mb/size_mb:.0f}% of fp32)")

    # Verify inference
    import onnxruntime as ort
    print(f"\nVerifying inference...")
    sess = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])

    test_input = np.random.randn(1, WINDOW_SAMPLES).astype(np.float32)
    output = sess.run(["logit"], {"input_values": test_input})
    print(f"  Single sample: output shape={output[0].shape}, value={output[0][0]:.4f}")

    # Benchmark
    test_batch = np.random.randn(8, WINDOW_SAMPLES).astype(np.float32)
    for _ in range(3):
        sess.run(["logit"], {"input_values": test_batch})

    t0 = time.perf_counter()
    for _ in range(20):
        sess.run(["logit"], {"input_values": test_batch})
    elapsed = (time.perf_counter() - t0) / 20

    ms_per_window = elapsed * 1000 / 8
    print(f"  Batch-8 latency: {elapsed*1000:.0f} ms ({ms_per_window:.0f} ms/window)")
    print(f"  ~{60/5 * ms_per_window / 1000:.1f}s per 1-min file")

    print(f"\nDone! Files:")
    print(f"  {onnx_path} — FP32 ({size_mb:.0f} MB)")
    print(f"  {quant_path} — INT8 ({quant_mb:.0f} MB)")


if __name__ == "__main__":
    main()
