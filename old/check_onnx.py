import onnxruntime as ort
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/speech_classifier_wav2vec2_7_5sec_quant.onnx"

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print(f"Model: {model_path}")
print("\nINPUTS:")
for inp in sess.get_inputs():
    print(f"  name:  {inp.name}")
    print(f"  shape: {inp.shape}")
    print(f"  type:  {inp.type}")

print("\nOUTPUTS:")
for out in sess.get_outputs():
    print(f"  name:  {out.name}")
    print(f"  shape: {out.shape}")
    print(f"  type:  {out.type}")