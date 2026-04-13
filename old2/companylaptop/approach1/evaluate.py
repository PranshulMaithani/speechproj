"""
Step 5: Detailed evaluation and error analysis.

Loads predictions, compares with ground truth, generates:
- Per-file error analysis with transcript excerpts
- Feature distribution comparison (cheating vs genuine)
- Threshold sweep for precision/recall trade-off

Usage:
    python -m approach1.evaluate
    python -m approach1.evaluate --predictions outputs/approach1/predictions.json
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve,
)

from approach1.config import FEATURES_CSV, MODEL_DIR, OUTPUT_DIR


def threshold_sweep(y_true, y_prob, thresholds=None):
    """Show precision/recall at different thresholds."""
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    print("\n  Threshold Sweep (cheating class):")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Flagged':>10}")
    print(f"  {'-'*50}")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        flagged = y_pred.sum()

        print(f"  {t:>10.2f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {flagged:>10}")


def error_analysis(df: pd.DataFrame, y_true, y_pred, y_prob):
    """Detailed error analysis: which files are misclassified and why."""
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            row = df.iloc[i]
            errors.append({
                "filename": row.get("filename", f"row_{i}"),
                "true_label": "CHEATING" if y_true[i] == 1 else "GENUINE",
                "predicted": "CHEATING" if y_pred[i] == 1 else "GENUINE",
                "probability": round(float(y_prob[i]), 4),
                "transcript": str(row.get("transcript", ""))[:200],
                # Key features
                "type_token_ratio": round(float(row.get("text__type_token_ratio", 0)), 4),
                "filler_rate": round(float(row.get("text__filler_rate", 0)), 4),
                "complex_word_rate": round(float(row.get("text__complex_word_rate", 0)), 4),
                "self_reference_rate": round(float(row.get("text__self_reference_rate", 0)), 4),
            })

    if not errors:
        print("\n  No misclassifications!")
        return errors

    # Categorize errors
    false_positives = [e for e in errors if e["true_label"] == "GENUINE"]
    false_negatives = [e for e in errors if e["true_label"] == "CHEATING"]

    print(f"\n  Error Analysis: {len(errors)} misclassifications")
    print(f"  False Positives (genuine flagged as cheating): {len(false_positives)}")
    print(f"  False Negatives (cheating missed): {len(false_negatives)}")

    if false_positives:
        print(f"\n  --- False Positives (worst 5) ---")
        for e in sorted(false_positives, key=lambda x: -x["probability"])[:5]:
            print(f"    {e['filename']}: P(cheat)={e['probability']:.3f}")
            print(f"      TTR={e['type_token_ratio']:.3f} filler={e['filler_rate']:.3f} "
                  f"complex={e['complex_word_rate']:.3f} self_ref={e['self_reference_rate']:.3f}")
            print(f"      \"{e['transcript'][:100]}...\"")

    if false_negatives:
        print(f"\n  --- False Negatives (worst 5) ---")
        for e in sorted(false_negatives, key=lambda x: x["probability"])[:5]:
            print(f"    {e['filename']}: P(cheat)={e['probability']:.3f}")
            print(f"      TTR={e['type_token_ratio']:.3f} filler={e['filler_rate']:.3f} "
                  f"complex={e['complex_word_rate']:.3f} self_ref={e['self_reference_rate']:.3f}")
            print(f"      \"{e['transcript'][:100]}...\"")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Evaluate cheating detection model")
    parser.add_argument("--model", default=None, help="Path to model .pkl")
    args = parser.parse_args()

    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run extract_features first.")
        return

    # Load features and model
    df = pd.read_csv(FEATURES_CSV)

    # Find model
    model_path = Path(args.model) if args.model else None
    if model_path is None:
        for tag in ["combined", "text", "acoustic"]:
            candidate = MODEL_DIR / f"xgboost_{tag}.pkl"
            if candidate.exists():
                model_path = candidate
                break
    if model_path is None or not model_path.exists():
        print("ERROR: No trained model found. Run training first.")
        return

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    print(f"Loaded model from {model_path}")

    # Prepare data
    X = df[feature_names].fillna(0).values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = df["label"].values.astype(int)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # ── Overall metrics ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FULL DATASET EVALUATION")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=["Genuine", "Cheating"]))

    if len(set(y_true)) > 1:
        print(f"  ROC AUC: {roc_auc_score(y_true, y_prob):.4f}")

    # ── Threshold sweep ───────────────────────────────────────────────
    threshold_sweep(y_true, y_prob)

    # ── Error analysis ────────────────────────────────────────────────
    errors = error_analysis(df, y_true, y_pred, y_prob)

    # ── Save error report ─────────────────────────────────────────────
    if errors:
        err_path = OUTPUT_DIR / "error_analysis.json"
        err_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Error details saved to {err_path}")

    # ── Feature distribution comparison ───────────────────────────────
    print("\n  Feature means by class:")
    print(f"  {'Feature':<40} {'Genuine':>10} {'Cheating':>10} {'Diff':>10}")
    print(f"  {'-'*70}")

    text_feats = [f for f in feature_names if f.startswith("text__")]
    for feat in text_feats[:15]:
        mean_0 = df.loc[df["label"] == 0, feat].mean()
        mean_1 = df.loc[df["label"] == 1, feat].mean()
        diff = mean_1 - mean_0
        print(f"  {feat:<40} {mean_0:>10.4f} {mean_1:>10.4f} {diff:>+10.4f}")


if __name__ == "__main__":
    main()
