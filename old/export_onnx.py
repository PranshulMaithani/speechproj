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
from typing import Dict, List

import yaml
import torch
import numpy as np


def _checkpoint_to_tag(checkpoint_name: str) -> str:
    """Convert checkpoint filename to a short artifact tag."""
    return checkpoint_name.replace("_best.pt", "")


def _infer_window_sec(tag: str, default_window_sec: float) -> float:
    """Infer window size from model tag, falling back to config default."""
    mapping = {
        "wav2vec2_7sec": 7.5,
        "wav2vec2_10sec": 10.0,
        "wav2vec2_12_5sec": 12.5,
        "wav2vec2_15sec": 15.0,
    }
    return mapping.get(tag, float(default_window_sec))


def _find_checkpoints(ckpt_dir: Path) -> List[Path]:
    """Return all best checkpoints sorted by name."""
    checkpoints = sorted(
        p for p in ckpt_dir.glob("*_best.pt")
        if p.is_file()
    )
    return checkpoints

def _benchmark_onnx(quant_path: Path, window_samples: int):
    """Benchmark quantized ONNX model on CPU."""
    import onnxruntime as ort
    import time

    sess = ort.InferenceSession(
        str(quant_path),
        providers=["CPUExecutionProvider"],
    )
    test_input = np.random.randn(8, window_samples).astype(np.float32)  # batch=8

    for _ in range(3):
        sess.run(["logits"], {"input_values": test_input})

    t0 = time.perf_counter()
    n_runs = 20
    for _ in range(n_runs):
        sess.run(["logits"], {"input_values": test_input})
    elapsed = (time.perf_counter() - t0) / n_runs

    ms_per_batch = elapsed * 1000
    ms_per_window = ms_per_batch / 8
    windows_per_min_file = 25
    sec_per_file = ms_per_window * windows_per_min_file / 1000

    print(f"  Batch-8 inference: {ms_per_batch:.1f}ms  ({ms_per_window:.1f}ms / window)")
    print(f"  Estimated time per 1-min file: {sec_per_file:.1f}s")
    print(f"  Estimated time for 475 files:  {475 * sec_per_file / 60:.1f} min")


def export_onnx(cfg: dict):
    data_root = Path(cfg["paths"]["data_root"])
    ckpt_dir = data_root / cfg["paths"]["checkpoints_dir"]
    w2v_cfg = cfg["training"]["wav2vec2"]

    device = torch.device("cpu")  # Export on CPU for portability

    # Load model class used by all variants
    from src.models.train_wavLM_5sec import SpeechClassifier

    checkpoints = _find_checkpoints(ckpt_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir} matching wav2vec2*_best.pt")

    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")

    sr = cfg["audio"]["sample_rate"]
    export_summary: Dict[str, Dict[str, str]] = {}

    for ckpt_path in checkpoints:
        tag = _checkpoint_to_tag(ckpt_path.name)
        window_sec = _infer_window_sec(tag, cfg["audio"]["window_sec"])
        window_samples = int(window_sec * sr)
        dummy_input = torch.zeros(1, window_samples)

        print("\n" + "=" * 70)
        print(f"Processing {ckpt_path.name}")
        print(f"  Window: {window_sec}s ({window_samples} samples @ {sr}Hz)")

        model = _get_classifier(ckpt_path, w2v_cfg)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        val_f1 = checkpoint.get("val_f1", None)
        val_f1_str = f"{val_f1:.4f}" if isinstance(val_f1, (int, float)) else "?"
        print(f"  Model loaded (epoch {checkpoint.get('epoch', '?')}, val_f1={val_f1_str})")

        # ------ Step 1: Export to ONNX (fp32) ------
        onnx_path = ckpt_dir / f"speech_classifier_{tag}.onnx"
        print(f"  Exporting to ONNX fp32: {onnx_path}")

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
            dynamo=False,
        )

        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ONNX model validated OK")
        print(f"  Size: {onnx_path.stat().st_size / 1e6:.1f} MB")

        # ------ Step 2: INT8 dynamic quantization ------
        quant_path = ckpt_dir / f"speech_classifier_{tag}_quant.onnx"
        print(f"  Quantizing to INT8: {quant_path}")

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantize_dynamic(
                str(onnx_path),
                str(quant_path),
                weight_type=QuantType.QInt8,
                nodes_to_exclude=[],
                op_types_to_quantize=["MatMul", "Gemm"],
            )
            print(f"  Quantized size: {quant_path.stat().st_size / 1e6:.1f} MB")

            import onnxruntime as ort
            sess = ort.InferenceSession(
                str(quant_path),
                providers=["CPUExecutionProvider"],
            )
            test_input = np.random.randn(1, window_samples).astype(np.float32)
            output = sess.run(["logits"], {"input_values": test_input})
            print(f"  Quantized test inference OK - shape: {output[0].shape}")

        except ImportError:
            print("  onnxruntime not installed, skipping quantization.")
            print("  Install with: pip install onnxruntime onnxruntime-tools")
            quant_path = onnx_path

        # ------ Step 3: Benchmark on CPU ------
        print("  Benchmarking CPU inference speed...")
        try:
            _benchmark_onnx(quant_path, window_samples)
        except Exception as e:
            print(f"  Benchmark skipped: {e}")

        export_summary[tag] = {
            "checkpoint": str(ckpt_path.name),
            "onnx": str(onnx_path.name),
            "quant": str(quant_path.name),
            "window_sec": str(window_sec),
        }

    print("\n" + "=" * 70)
    print("Export complete for all models")
    for tag, paths in export_summary.items():
        print(f"  [{tag}] fp32={paths['onnx']} | int8={paths['quant']} | window={paths['window_sec']}s")
    print(f"\nAlso copy: {ckpt_dir / 'xgboost_baseline.json'}")
    print(f"Also copy: {ckpt_dir / 'xgboost_scaler.pkl'}")

def _get_classifier(ckpt_path: Path, w2v_cfg: dict):
    """Load the correct SpeechClassifier class based on checkpoint name."""
    if "wavlm" in ckpt_path.name.lower():
        from src.models.train_wavLM_5sec import SpeechClassifier
    else:
        from src.models.train_wav2vec2_5sec import SpeechClassifier
    return SpeechClassifier(
        model_name=w2v_cfg["model_name"],
        hidden_size=w2v_cfg["hidden_size"],
        num_labels=2,
        dropout=0.0,
        freeze_layers=0,
    )
    
def _infer_window_sec(tag: str, default_window_sec: float) -> float:
    mapping = {
        "wav2vec2_7sec": 7.5,
        "wav2vec2_10sec": 10.0,
        "wav2vec2_12_5sec": 12.5,
        "wav2vec2_15sec": 15.0,
        "wavlm_5sec": 5.0,       # add this
    }
    return mapping.get(tag, float(default_window_sec))

def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    export_onnx(cfg)


if __name__ == "__main__":
    main()
