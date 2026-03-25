"""
Step 3: Train XGBoost classifier on extracted features.

Trains multiple variants:
  - Text-only features (validates linguistic signal)
  - Acoustic-only features (validates acoustic signal)
  - Combined text + acoustic (full model)

Uses stratified train/val/test split. Reports per-feature importance.

Usage:
    python -m approach1.train
    python -m approach1.train --text-only    # quick validation of linguistic signal
    python -m approach1.train --no-search    # skip hyperparameter search
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from approach1.config import (
    FEATURES_CSV, MODEL_DIR, OUTPUT_DIR, TEST_RATIO, VAL_RATIO, RANDOM_SEED,
)


def load_and_split(features_csv: Path, feature_prefix: str = None):
    """Load features CSV and split into train/val/test."""
    df = pd.read_csv(features_csv)

    # Select feature columns
    if feature_prefix == "text":
        feat_cols = [c for c in df.columns if c.startswith("text__")]
    elif feature_prefix == "acoustic":
        feat_cols = [c for c in df.columns if c.startswith("acoustic__")]
    else:
        feat_cols = [c for c in df.columns if c.startswith("text__") or c.startswith("acoustic__")]

    if not feat_cols:
        raise ValueError(f"No feature columns found with prefix '{feature_prefix}'. "
                         f"Available columns: {[c for c in df.columns if '__' in c][:10]}")

    print(f"  Using {len(feat_cols)} features ({feature_prefix or 'all'})")

    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df["label"].values.astype(int)
    filenames = df["filename"].values

    # Replace inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Stratified split: test first, then train/val
    np.random.seed(RANDOM_SEED)
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)

    # Stratified split
    from sklearn.model_selection import train_test_split
    idx_trainval, idx_test = train_test_split(
        indices, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y[indices],
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=VAL_RATIO / (1 - TEST_RATIO),
        random_state=RANDOM_SEED, stratify=y[idx_trainval],
    )

    return {
        "X_train": X[idx_train], "y_train": y[idx_train], "files_train": filenames[idx_train],
        "X_val": X[idx_val], "y_val": y[idx_val], "files_val": filenames[idx_val],
        "X_test": X[idx_test], "y_test": y[idx_test], "files_test": filenames[idx_test],
        "feature_names": feat_cols,
        "df": df,
    }


def train_logistic(data: dict) -> dict:
    """Train logistic regression baseline."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data["X_train"])
    X_val = scaler.transform(data["X_val"])
    X_test = scaler.transform(data["X_test"])

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    model.fit(X_train, data["y_train"])

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "scaler": scaler,
        "val_acc": accuracy_score(data["y_val"], y_val_pred),
        "val_f1": f1_score(data["y_val"], y_val_pred),
        "test_acc": accuracy_score(data["y_test"], y_test_pred),
        "test_precision": precision_score(data["y_test"], y_test_pred, zero_division=0),
        "test_recall": recall_score(data["y_test"], y_test_pred, zero_division=0),
        "test_f1": f1_score(data["y_test"], y_test_pred, zero_division=0),
        "test_auc": roc_auc_score(data["y_test"], y_test_prob) if len(set(data["y_test"])) > 1 else 0,
        "test_preds": y_test_pred,
        "test_probs": y_test_prob,
    }


def train_xgboost(data: dict, do_search: bool = True) -> dict:
    """Train XGBoost with optional hyperparameter search."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    if do_search:
        print("  Searching hyperparameters (5-fold CV)...")
        best_f1 = 0
        best_params = {}

        param_grid = [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": ss, "colsample_bytree": cs}
            for n in [100, 300, 500]
            for d in [3, 5, 7]
            for lr in [0.01, 0.05, 0.1]
            for ss in [0.8, 1.0]
            for cs in [0.7, 1.0]
        ]

        # Sample a reasonable number of combinations
        np.random.seed(RANDOM_SEED)
        if len(param_grid) > 30:
            param_grid = [param_grid[i] for i in np.random.choice(len(param_grid), 30, replace=False)]

        for params in param_grid:
            model = xgb.XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                verbosity=0,
            )
            # Use cross-validation on train set
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            cv_preds = cross_val_predict(model, X_train, y_train, cv=skf)
            cv_f1 = f1_score(y_train, cv_preds)

            if cv_f1 > best_f1:
                best_f1 = cv_f1
                best_params = params

        print(f"  Best CV F1: {best_f1:.4f}")
        print(f"  Best params: {best_params}")
    else:
        best_params = {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }

    # Train final model
    model = xgb.XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        verbosity=0,
        early_stopping_rounds=20,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    X_test = data["X_test"]
    y_test = data["y_test"]
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Feature importance
    importance = dict(zip(data["feature_names"], model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    return {
        "model": model,
        "params": best_params,
        "val_acc": accuracy_score(y_val, model.predict(X_val)),
        "val_f1": f1_score(y_val, model.predict(X_val)),
        "test_acc": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
        "test_auc": roc_auc_score(y_test, y_test_prob) if len(set(y_test)) > 1 else 0,
        "test_preds": y_test_pred,
        "test_probs": y_test_prob,
        "feature_importance": importance,
    }


def print_results(name: str, results: dict, data: dict):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Val  Accuracy: {results['val_acc']:.4f}  |  F1: {results['val_f1']:.4f}")
    print(f"  Test Accuracy: {results['test_acc']:.4f}  |  F1: {results['test_f1']:.4f}")
    print(f"  Test Precision: {results['test_precision']:.4f}")
    print(f"  Test Recall:    {results['test_recall']:.4f}")
    print(f"  Test AUC:       {results['test_auc']:.4f}")
    print()
    print("  Confusion Matrix (test set):")
    cm = confusion_matrix(data["y_test"], results["test_preds"])
    print(f"    {'':>12} Pred:Genuine  Pred:Cheat")
    print(f"    {'True:Genuine':>12}  {cm[0][0]:>8}  {cm[0][1]:>8}")
    print(f"    {'True:Cheat':>12}  {cm[1][0]:>8}  {cm[1][1]:>8}")
    print()

    if "feature_importance" in results:
        print("  Top 15 features:")
        for i, (feat, imp) in enumerate(list(results["feature_importance"].items())[:15]):
            bar = "#" * int(imp * 200)
            print(f"    {i+1:>2}. {feat:<40} {imp:.4f} {bar}")


def main():
    parser = argparse.ArgumentParser(description="Train cheating detection classifier")
    parser.add_argument("--text-only", action="store_true", help="Train on text features only")
    parser.add_argument("--acoustic-only", action="store_true", help="Train on acoustic features only")
    parser.add_argument("--no-search", action="store_true", help="Skip hyperparameter search")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURES_CSV.exists():
        print(f"ERROR: Features file not found at {FEATURES_CSV}")
        print("Run step 2 first: python -m approach1.extract_features")
        return

    # ── Determine which feature sets to train ─────────────────────────
    experiments = []
    if args.text_only:
        experiments = [("text", "Text-Only (Linguistic Signal Validation)")]
    elif args.acoustic_only:
        experiments = [("acoustic", "Acoustic-Only")]
    else:
        experiments = [
            ("text", "Text-Only (Linguistic Signal)"),
            ("acoustic", "Acoustic-Only"),
            (None, "Combined Text + Acoustic"),
        ]

    all_results = {}

    for prefix, name in experiments:
        print(f"\n{'#'*60}")
        print(f"# Experiment: {name}")
        print(f"{'#'*60}")

        data = load_and_split(FEATURES_CSV, feature_prefix=prefix)
        print(f"  Train: {len(data['y_train'])} ({data['y_train'].sum()} cheat)")
        print(f"  Val:   {len(data['y_val'])} ({data['y_val'].sum()} cheat)")
        print(f"  Test:  {len(data['y_test'])} ({data['y_test'].sum()} cheat)")

        # Logistic Regression baseline
        print("\n  Training Logistic Regression baseline...")
        lr_results = train_logistic(data)
        print_results(f"LogisticRegression - {name}", lr_results, data)

        # XGBoost
        print(f"\n  Training XGBoost...")
        xgb_results = train_xgboost(data, do_search=not args.no_search)
        print_results(f"XGBoost - {name}", xgb_results, data)

        # Save best model
        tag = prefix or "combined"
        model_path = MODEL_DIR / f"xgboost_{tag}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": xgb_results["model"], "feature_names": data["feature_names"]}, f)
        print(f"  Model saved to {model_path}")

        all_results[tag] = {
            "logistic": {k: v for k, v in lr_results.items() if k not in ("model", "scaler", "test_preds", "test_probs")},
            "xgboost": {k: v for k, v in xgb_results.items() if k not in ("model", "test_preds", "test_probs", "feature_importance")},
            "feature_importance_top20": dict(list(xgb_results.get("feature_importance", {}).items())[:20]),
        }

    # ── Save summary ──────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "training_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults summary saved to {summary_path}")

    # ── Final recommendation ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for tag, res in all_results.items():
        xgb_r = res["xgboost"]
        print(f"  {tag:>12}: Precision={xgb_r['test_precision']:.3f}  Recall={xgb_r['test_recall']:.3f}  "
              f"F1={xgb_r['test_f1']:.3f}  AUC={xgb_r['test_auc']:.3f}")

    if "text" in all_results:
        text_f1 = all_results["text"]["xgboost"]["test_f1"]
        if text_f1 > 0.60:
            print(f"\n  >>> LINGUISTIC SIGNAL CONFIRMED (text-only F1 = {text_f1:.3f})")
            print(f"  >>> Proceed with full pipeline (combined features)")
        else:
            print(f"\n  >>> Linguistic signal WEAK (text-only F1 = {text_f1:.3f})")
            print(f"  >>> Consider pivoting to neural approach (Whisper encoder fine-tuning)")


if __name__ == "__main__":
    main()
