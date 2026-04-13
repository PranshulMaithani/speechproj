"""
XGBoost Baseline Trainer
==========================
Trains an XGBoost classifier on handcrafted prosodic features.
Provides an interpretable baseline and feature importance analysis.

Usage:
    python -m src.models.train_xgboost --config configs/config.yaml
"""

import argparse
import json
import pickle
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def load_features(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Load pre-extracted features and identify feature columns."""
    data_root = Path(cfg["paths"]["data_root"])
    feat_path = data_root / cfg["paths"]["features_dir"] / "window_features.csv"
    df = pd.read_csv(feat_path)

    meta_cols = [
        "source_file", "speaker_id", "l1", "gender", "task",
        "label", "label_int", "split", "window_idx",
        "window_start_sec", "window_end_sec", "speech_ratio",
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Replace inf/nan
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    return df, feature_cols


def compute_sample_weights(df: pd.DataFrame, accent_weights: dict) -> np.ndarray:
    """Compute per-sample weights based on accent oversampling config."""
    weights = df["l1"].map(accent_weights).fillna(1.0).values

    # Also balance classes
    label_counts = df["label_int"].value_counts()
    total = len(df)
    class_weights = {
        label: total / (2 * count) for label, count in label_counts.items()
    }
    class_w = df["label_int"].map(class_weights).values

    return weights * class_w


def train_xgboost(cfg: dict):
    """Train and evaluate XGBoost classifier."""
    data_root = Path(cfg["paths"]["data_root"])
    xgb_cfg = cfg["training"]["xgb"]
    accent_weights = cfg.get("accent_weights", {})

    print("Loading features...")
    df, feature_cols = load_features(cfg)

    # Split data
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label_int"].values
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["label_int"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label_int"].values

    print(f"Train: {len(X_train)} windows | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")

    # Sample weights for accent oversampling
    sample_weights = compute_sample_weights(train_df, accent_weights)

    # Scale features (XGBoost doesn't require it, but helps stability)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train XGBoost
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        tree_method="hist",  # Fast training
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    results = {}
    for split_name, X, y, split_df in [
        ("val", X_val, y_val, val_df),
        ("test", X_test, y_test, test_df),
    ]:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.0

        results[split_name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4),
        }

        print(f"\n--- {split_name.upper()} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"AUC:       {auc:.4f}")
        print(f"\n{classification_report(y, preds, target_names=['spontaneous', 'read'])}")

        # Per-accent breakdown
        print(f"Per-accent accuracy ({split_name}):")
        for l1, grp in split_df.groupby("l1"):
            mask = grp.index.isin(split_df.index)
            grp_indices = [list(split_df.index).index(i) for i in grp.index if i in split_df.index]
            if len(grp_indices) > 0:
                grp_preds = preds[grp_indices]
                grp_true = y[grp_indices]
                grp_acc = accuracy_score(grp_true, grp_preds)
                print(f"  {l1:4s}: {grp_acc:.3f} ({len(grp_indices)} windows)")

    # Save model & artifacts
    ckpt_dir = data_root / cfg["paths"]["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_path = ckpt_dir / "xgboost_baseline.json"
    model.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")

    scaler_path = ckpt_dir / "xgboost_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {scaler_path}")

    results_path = ckpt_dir / "xgboost_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Feature importance plot
    plot_feature_importance(model, feature_cols, ckpt_dir)

    # Confusion matrix plot
    plot_confusion_matrix(y_test, model.predict(X_test), ckpt_dir)

    return model, scaler, results


def plot_feature_importance(model, feature_names, save_dir):
    """Plot and save top-30 feature importances."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[-30:]

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(range(len(indices)), importance[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top 30 Features — XGBoost Baseline")
    plt.tight_layout()
    plt.savefig(save_dir / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Feature importance plot saved: {save_dir / 'feature_importance.png'}")


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Spontaneous", "Read"],
        yticklabels=["Spontaneous", "Read"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost baseline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_xgboost(cfg)


if __name__ == "__main__":
    main()
