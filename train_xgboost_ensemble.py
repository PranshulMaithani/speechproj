"""
Train XGBoost Ensemble on Multi-Signal Features
=================================================
Combines text features + pause features + prosodic features + acoustic model scores
to classify read/prepared vs spontaneous speech.

Usage:
    # Train on extracted features
    python train_xgboost_ensemble.py --features datasets/features_combined.csv

    # Train on multiple feature files
    python train_xgboost_ensemble.py --features datasets/features_allsstar.csv datasets/features_libri.csv datasets/features_ami.csv

    # With acoustic model scores
    python train_xgboost_ensemble.py --features datasets/features_combined.csv --acoustic-scores datasets/acoustic_scores.csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import joblib

OUTPUT_DIR = Path("checkpoints_ensemble")

# Feature columns by group
TEXT_FEATURES = [
    "filler_rate", "filler_count", "repetition_rate", "repair_rate",
    "ttr", "mattr", "complex_word_rate", "avg_word_length",
    "n_words", "n_unique_words",
    "avg_sentence_length", "std_sentence_length", "fragment_rate", "n_sentences",
    "self_ref_rate", "discourse_marker_rate", "hedge_rate",
    "noun_rate", "verb_rate", "adj_rate",
]

PAUSE_FEATURES = [
    "pause_mean", "pause_std", "pause_median", "pause_skew",
    "long_pause_rate", "pause_ratio", "n_pauses", "pause_regularity",
    "pause_before_content_ratio", "pause_before_function_ratio",
    "mid_phrase_pause_rate", "words_per_sec", "articulation_rate",
]

PROSODIC_FEATURES = [
    "f0_mean", "f0_std", "f0_range", "f0_skew", "f0_slope",
    "energy_mean", "energy_std", "speaking_rate_std",
]

WAV2VEC2_FEATURES = [
    "wav2vec2_read_ratio", "wav2vec2_mean_p_read", "wav2vec2_max_p_read",
]

ALL_FEATURES = TEXT_FEATURES + PAUSE_FEATURES + PROSODIC_FEATURES + WAV2VEC2_FEATURES


def load_and_combine_features(feature_paths, acoustic_path=None):
    """Load feature CSVs and combine."""
    dfs = []
    for path in feature_paths:
        df = pd.read_csv(path)
        print(f"  Loaded {path}: {len(df)} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Filter out rows without labels
    combined = combined[combined["label_int"].isin([0, 1])].reset_index(drop=True)

    # Add acoustic scores if available
    if acoustic_path and os.path.exists(acoustic_path):
        acoustic_df = pd.read_csv(acoustic_path)
        combined = combined.merge(acoustic_df[["filepath", "acoustic_p_read"]],
                                   on="filepath", how="left")
        combined["acoustic_p_read"] = combined["acoustic_p_read"].fillna(0.5)
        print(f"  Merged acoustic scores from {acoustic_path}")

    print(f"\nCombined: {len(combined)} samples")
    print(f"  Read:        {(combined['label_int']==1).sum()}")
    print(f"  Spontaneous: {(combined['label_int']==0).sum()}")

    return combined


def get_available_features(df):
    """Return feature columns that exist and have non-zero variance."""
    available = []
    for col in ALL_FEATURES:
        if col in df.columns:
            vals = df[col].fillna(0)
            if vals.std() > 1e-8:
                available.append(col)
    return available


def train(df, feature_cols, output_dir):
    """Train XGBoost with cross-validation and feature importance analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_cols].fillna(0).values
    y = df["label_int"].values

    # Split: use existing splits if available, otherwise random
    if "split" in df.columns and df["split"].nunique() > 1:
        train_mask = df["split"] == "train"
        val_mask = df["split"] == "val"
        test_mask = df["split"] == "test"

        # If no test split, create from train
        if test_mask.sum() == 0:
            train_idx = df[train_mask].index
            t_idx, test_idx = train_test_split(train_idx, test_size=0.15, random_state=42,
                                                stratify=y[train_idx])
            train_mask = df.index.isin(t_idx)
            test_mask = df.index.isin(test_idx)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val = X[val_mask] if val_mask.sum() > 0 else None
        y_val = y[val_mask] if val_mask.sum() > 0 else None
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
            X, y, np.arange(len(y)), test_size=0.15, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_train_val, y_train_val, idx_train_val, test_size=0.18, random_state=42, stratify=y_train_val
        )
        train_mask = np.isin(np.arange(len(y)), idx_train)
        test_mask = np.isin(np.arange(len(y)), idx_test)

    print(f"\nTrain: {len(y_train)} ({y_train.sum()} read, {(1-y_train).sum():.0f} spont)")
    if X_val is not None:
        print(f"Val:   {len(y_val)} ({y_val.sum()} read, {(1-y_val).sum():.0f} spont)")
    print(f"Test:  {len(y_test)} ({y_test.sum()} read, {(1-y_test).sum():.0f} spont)")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # XGBoost
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        use_label_encoder=False,
    )

    eval_set = [(X_train_s, y_train)]
    if X_val is not None:
        X_val_s = scaler.transform(X_val)
        eval_set.append((X_val_s, y_val))

    print("\nTraining XGBoost...")
    model.fit(X_train_s, y_train, eval_set=eval_set, verbose=50)

    # Cross-validation score
    print("\n5-fold cross-validation on train set...")
    cv_scores = cross_val_score(
        xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                           use_label_encoder=False, eval_metric="logloss"),
        X_train_s, y_train, cv=5, scoring="f1"
    )
    print(f"  CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Test evaluation
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['spontaneous', 'read'])}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Feature importance
    print(f"\nFeature Importance (top 20):")
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    for name, imp in feat_imp[:20]:
        bar = "#" * int(imp * 100)
        print(f"  {name:<35s} {imp:.4f} {bar}")

    # Threshold analysis
    print(f"\nThreshold Sensitivity:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        t_pred = (y_proba >= thresh).astype(int)
        t_prec = precision_score(y_test, t_pred, zero_division=0)
        t_rec = recall_score(y_test, t_pred, zero_division=0)
        t_f1 = f1_score(y_test, t_pred, zero_division=0)
        marker = " <-- default" if thresh == 0.5 else ""
        print(f"  threshold={thresh:.1f}: prec={t_prec:.4f} rec={t_rec:.4f} f1={t_f1:.4f}{marker}")

    # Feature group ablation
    print(f"\nFeature Group Ablation:")
    groups = {
        "Text only": [c for c in feature_cols if c in TEXT_FEATURES],
        "Pause only": [c for c in feature_cols if c in PAUSE_FEATURES],
        "Prosodic only": [c for c in feature_cols if c in PROSODIC_FEATURES],
        "Text + Pause": [c for c in feature_cols if c in TEXT_FEATURES + PAUSE_FEATURES],
        "All features": feature_cols,
    }
    for name, cols in groups.items():
        if not cols:
            continue
        X_abl = df[cols].fillna(0).values
        abl_scaler = StandardScaler()
        X_tr = abl_scaler.fit_transform(X_abl[train_mask])
        X_te = abl_scaler.transform(X_abl[test_mask])
        abl_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                        use_label_encoder=False, eval_metric="logloss")
        abl_model.fit(X_tr, y_train, verbose=False)
        abl_pred = abl_model.predict(X_te)
        abl_f1 = f1_score(y_test, abl_pred)
        print(f"  {name:<20s} ({len(cols):2d} features): F1={abl_f1:.4f}")

    # Save model and artifacts
    model.save_model(str(output_dir / "xgboost_ensemble.json"))
    joblib.dump(scaler, str(output_dir / "scaler.pkl"))

    results = {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "auc": round(auc, 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "feature_importances": {name: round(float(imp), 6) for name, imp in feat_imp},
    }
    with open(output_dir / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to: {output_dir}")
    print(f"  xgboost_ensemble.json  — XGBoost model")
    print(f"  scaler.pkl             — Feature scaler")
    print(f"  ensemble_results.json  — Results + feature importances")

    return model, scaler, results


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost ensemble")
    parser.add_argument("--features", nargs="+", required=True, help="Feature CSV files")
    parser.add_argument("--acoustic-scores", type=str, help="Acoustic model scores CSV")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--no-wav2vec2", action="store_true", help="Exclude wav2vec2 features (text-only model)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load features
    df = load_and_combine_features(args.features, args.acoustic_scores)

    # Get available features
    feature_cols = get_available_features(df)
    if args.no_wav2vec2:
        feature_cols = [c for c in feature_cols if c not in WAV2VEC2_FEATURES]
        print("\n** Excluding wav2vec2 features (text-only mode) **")
    print(f"\nUsing {len(feature_cols)} features:")
    for col in feature_cols:
        group = "TEXT" if col in TEXT_FEATURES else "PAUSE" if col in PAUSE_FEATURES else "PROSODIC" if col in PROSODIC_FEATURES else "WAV2VEC2" if col in WAV2VEC2_FEATURES else "OTHER"
        print(f"  [{group:8s}] {col}")

    # Train
    train(df, feature_cols, output_dir)


if __name__ == "__main__":
    main()
