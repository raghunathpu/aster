"""
ASTER - Model Training
=======================
Trains a two-stage classification pipeline:
  Stage 1 - Baseline: Random Forest
  Stage 2 - Main:     Gradient Boosting (sklearn)

Target: impact_tier  ->  Low / Medium / High
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------------------------
# Add project root to path
# ----------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.preprocessing.data_loader import preprocess
from src.features.feature_engineering import build_features, encode_for_model

LABEL_ORDER = ["Low", "Medium", "High"]


def load_and_prepare(data_path: str):
    """Full pipeline from raw CSV to X, y."""
    df = preprocess(data_path)
    df = build_features(df)

    # Filter rows with valid target
    df = df[df["impact_tier"].isin(LABEL_ORDER)].copy()
    df = df[df["latitude"] > 1].copy()   # remove zero/null coords

    X, encoders = encode_for_model(df)
    y = df["impact_tier"]

    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    y_enc = le.transform(y)

    return X, y_enc, encoders, le, df


def compute_class_weights(y_enc, le):
    classes = np.unique(y_enc)
    weights = compute_class_weight("balanced", classes=classes, y=y_enc)
    return dict(zip(classes, weights))


def train_baseline(X_train, y_train, class_weight_map):
    """Random Forest baseline."""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight=class_weight_map,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf


def train_main_model(X_train, y_train, class_weight_map):
    """Gradient Boosting main model."""
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    return gb


def evaluate(model, X_test, y_test, le, label=""):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (macro)    : {f1_macro:.4f}")
    print(f"  F1 (weighted) : {f1_weighted:.4f}")
    print(f"  AUC-ROC (ovr) : {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
        "report": classification_report(y_test, y_pred, target_names=le.classes_),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def get_feature_importance(model, feature_names, top_n=25):
    """Extract top feature importances."""
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(top_n)
    return fi


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix - Impact Tier Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(fi, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#2563EB" if i < 5 else "#93C5FD" for i in range(len(fi))]
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("Top Features - ASTER Impact Tier Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def train_pipeline(data_path: str, model_dir: str, assets_dir: str):
    """End-to-end training pipeline. Saves model and artefacts."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    print("\n[1/6] Loading and preparing data...")
    X, y, encoders, le, df_full = load_and_prepare(data_path)
    print(f"  Dataset: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"  Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    print("\n[2/6] Train/validation split (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    class_weights = compute_class_weights(y_train, le)

    print("\n[3/6] Training Random Forest baseline...")
    rf = train_baseline(X_train, y_train, class_weights)
    rf_metrics = evaluate(rf, X_test, y_test, le, label="Random Forest Baseline")

    print("\n[4/6] Training Gradient Boosting main model...")
    gb = train_main_model(X_train, y_train, class_weights)
    gb_metrics = evaluate(gb, X_test, y_test, le, label="Gradient Boosting (Main)")

    print("\n[5/6] Cross-validation (5-fold)...")
    cv_scores = cross_val_score(gb, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                 scoring="f1_macro", n_jobs=-1)
    print(f"  CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    gb_metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
    gb_metrics["cv_f1_std"] = round(cv_scores.std(), 4)

    print("\n[6/6] Saving artefacts...")

    # Save models
    joblib.dump(rf, os.path.join(model_dir, "rf_baseline.pkl"))
    joblib.dump(gb, os.path.join(model_dir, "gb_main.pkl"))
    joblib.dump(encoders, os.path.join(model_dir, "encoders.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(list(X.columns), os.path.join(model_dir, "feature_names.pkl"))

    # Save feature importance
    fi = get_feature_importance(gb, list(X.columns))
    fi.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)

    # Save metrics
    metrics_all = {
        "random_forest": rf_metrics,
        "gradient_boosting": gb_metrics,
    }
    with open(os.path.join(model_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)

    # Plots
    cm = np.array(gb_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, le.classes_, os.path.join(assets_dir, "confusion_matrix.png"))
    plot_feature_importance(fi, os.path.join(assets_dir, "feature_importance.png"))

    # Save processed dataset for the app's EDA tab
    df_full.to_csv(os.path.join(model_dir, "processed_dataset.csv"), index=False)

    print("\n[OK] Training complete. All artefacts saved to:", model_dir)
    print(f"\n  RF  accuracy: {rf_metrics['accuracy']}")
    print(f"  GB  accuracy: {gb_metrics['accuracy']}")
    print(f"  GB  F1-macro: {gb_metrics['f1_macro']}")
    print(f"  GB  AUC-ROC:  {gb_metrics['auc_roc']}")

    return gb, encoders, le


if __name__ == "__main__":
    DATA_PATH = os.path.join(ROOT, "data", "bengaluru_traffic_events.csv")
    MODEL_DIR = os.path.join(ROOT, "models")
    ASSETS_DIR = os.path.join(ROOT, "assets")
    train_pipeline(DATA_PATH, MODEL_DIR, ASSETS_DIR)
