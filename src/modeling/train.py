"""
ASTER - Model Training
=======================
Trains a two-stage classification pipeline:
  Stage 1 - Baseline:    Random Forest
  Stage 2 - Main (Honest): Gradient Boosting WITHOUT leaky features
  Stage 3 - Demo:        Gradient Boosting WITH all features (for scoring rule validation only)

Target: impact_tier  ->  Low / Medium / High

NOTE ON LEAKAGE:
  The target variable (impact_tier) is computed from:
    road_closure_flag, priority, event_cause, corridor
  Including road_closure_flag and priority directly in the feature matrix
  creates a deterministic model (~99.9% accuracy) that simply re-learns the
  scoring rule — NOT a generalizable predictor.

  The HONEST model removes these features and achieves ~93% accuracy using
  only pre-dispatch observable inputs: location, cause, vehicle type, time,
  and historical frequency rates.
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.preprocessing.data_loader import preprocess
from src.features.feature_engineering import build_features, encode_for_model

LABEL_ORDER = ["Low", "Medium", "High"]

# ─────────────────────────────────────────────────────────────────────
# Features that MUST be excluded from honest model (they are computed
# from or directly encode parts of the target variable formula)
# ─────────────────────────────────────────────────────────────────────
LEAKY_FEATURES = {
    "road_closure_flag",    # directly in scoring formula (+2 pts)
    "priority",             # directly in scoring formula (+1 pt)
    # one-hot encodings of the above that encode_for_model creates:
    "priority_High", "priority_Low",
}


def load_data(data_path: str):
    df = preprocess(data_path)
    df = df[df["impact_tier"].isin(LABEL_ORDER)].copy()
    df = df[df["latitude"] > 1].copy()
    df = df.sort_values("start_local").reset_index(drop=True)
    return df


def compute_class_weights(y_enc, le):
    classes = np.unique(y_enc)
    weights = compute_class_weight("balanced", classes=classes, y=y_enc)
    return dict(zip(classes, weights))


def remove_leaky_cols(X: pd.DataFrame) -> pd.DataFrame:
    """Drop any column that leaks target information."""
    leaky_cols = [c for c in X.columns if c in LEAKY_FEATURES
                  or c.startswith("priority_")
                  or c == "road_closure_flag"]
    return X.drop(columns=leaky_cols, errors="ignore")


def train_baseline(X_train, y_train, class_weight_map):
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_leaf=5, class_weight=class_weight_map,
        n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf


def train_main_model(X_train, y_train, class_weight_map):
    """Honest GBM — no leaky features."""
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, min_samples_leaf=10,
        subsample=0.8, random_state=42,
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
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(top_n)
    return fi


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0F1729")
    ax.set_facecolor("#1E2D4E")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="#2D4070"
    )
    ax.set_xlabel("Predicted", fontsize=12, color="#E2E8F0")
    ax.set_ylabel("Actual", fontsize=12, color="#E2E8F0")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#E2E8F0")
    ax.tick_params(colors="#E2E8F0")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(fi, save_path, title="Top Features"):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0F1729")
    ax.set_facecolor("#1E2D4E")
    colors = ["#3B82F6" if i < 5 else "#93C5FD" for i in range(len(fi))]
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11, color="#E2E8F0")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#E2E8F0")
    ax.tick_params(colors="#E2E8F0")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#2D4070")
    ax.grid(axis="x", color="#2D4070", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


def train_pipeline(data_path: str, model_dir: str, assets_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    print("\n[1/7] Loading and splitting data (Time-Series)...")
    df_full = load_data(data_path)

    n_train = int(len(df_full) * 0.8)
    df_train = df_full.iloc[:n_train].copy()
    df_test = df_full.iloc[n_train:].copy()

    print(f"  Dataset: {len(df_full)} rows")
    print(f"  Train: {len(df_train)} | Test: {len(df_test)}")

    print("\n[2/7] Building features...")
    df_train, freq_maps = build_features(df_train)
    df_test, _ = build_features(df_test, freq_maps)
    df_full_feat, _ = build_features(df_full, freq_maps)

    X_train_all, encoders = encode_for_model(df_train)
    X_test_all, _ = encode_for_model(df_test, encoders)
    X_all, _ = encode_for_model(df_full_feat, encoders)

    # ── HONEST feature matrices (leaky cols removed) ─────────────
    X_train_honest = remove_leaky_cols(X_train_all)
    X_test_honest = remove_leaky_cols(X_test_all)
    X_honest = remove_leaky_cols(X_all)
    honest_fnames = list(X_train_honest.columns)

    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    y_train = le.transform(df_train["impact_tier"])
    y_test = le.transform(df_test["impact_tier"])
    y = le.transform(df_full_feat["impact_tier"])

    print(f"  Class distribution (Train): {dict(zip(le.classes_, np.bincount(y_train)))}")
    print(f"  Honest feature count: {len(honest_fnames)} (removed {len(list(X_train_all.columns)) - len(honest_fnames)} leaky features)")

    class_weights = compute_class_weights(y_train, le)

    print("\n[3/7] Training Random Forest baseline (all features)...")
    rf = train_baseline(X_train_all, y_train, class_weights)
    rf_metrics = evaluate(rf, X_test_all, y_test, le, label="Random Forest Baseline")

    print("\n[4/7] Training HONEST Gradient Boosting (pre-dispatch features only)...")
    gb = train_main_model(X_train_honest, y_train, class_weights)
    gb_metrics = evaluate(gb, X_test_honest, y_test, le, label="Gradient Boosting — HONEST (No Leakage)")

    print("\n[5/7] Training Scoring-Rule Validation model (all features — shows rule consistency)...")
    gb_full = train_main_model(X_train_all, y_train, class_weights)
    gb_full_metrics = evaluate(gb_full, X_test_all, y_test, le, label="GBM Scoring Rule Validator (ALL features)")
    gb_full_metrics["note"] = (
        "This model includes road_closure_flag and priority — features used in the target "
        "variable formula. The near-perfect accuracy validates that the scoring rule is "
        "internally consistent, NOT that the model is a generalizable predictor. "
        "The HONEST model (gb_main.pkl) is what ASTER uses for live inference."
    )

    print("\n[6/7] Cross-validation (Time-Series Split, HONEST model)...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(gb, X_honest, y, cv=tscv, scoring="f1_macro", n_jobs=-1)
    print(f"  CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    gb_metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
    gb_metrics["cv_f1_std"] = round(cv_scores.std(), 4)
    gb_metrics["honest"] = True
    gb_metrics["leaky_features_removed"] = list(set(X_train_all.columns) - set(honest_fnames))

    print("\n[7/7] Saving artefacts...")

    # Save HONEST model (used for live inference)
    joblib.dump(gb, os.path.join(model_dir, "gb_main.pkl"))
    joblib.dump(rf, os.path.join(model_dir, "rf_baseline.pkl"))
    joblib.dump(gb_full, os.path.join(model_dir, "gb_scoring_rule.pkl"))  # for transparency display
    joblib.dump(encoders, os.path.join(model_dir, "encoders.pkl"))
    joblib.dump(freq_maps, os.path.join(model_dir, "freq_maps.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(honest_fnames, os.path.join(model_dir, "feature_names.pkl"))

    # Save feature importance
    fi = get_feature_importance(gb, honest_fnames)
    fi.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)

    # Save metrics
    metrics_all = {
        "random_forest": rf_metrics,
        "gradient_boosting": gb_metrics,          # HONEST model
        "scoring_rule_validator": gb_full_metrics, # ALL features (transparency only)
    }
    with open(os.path.join(model_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)

    # Plots for honest model
    cm = np.array(gb_metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm, le.classes_,
        os.path.join(assets_dir, "confusion_matrix.png"),
        title="ASTER Confusion Matrix — Honest Model (Pre-Dispatch Features Only)"
    )
    plot_feature_importance(fi, os.path.join(assets_dir, "feature_importance.png"),
                            title="Top Features — Honest GBM (No Leaky Features)")

    # Save processed dataset
    df_full_feat.to_csv(os.path.join(model_dir, "processed_dataset.csv"), index=False)

    print("\n[OK] Training complete.")
    print(f"\n  RF  accuracy       : {rf_metrics['accuracy']}")
    print(f"  GB  HONEST acc     : {gb_metrics['accuracy']}  <-- USE THIS")
    print(f"  GB  Scoring-rule   : {gb_full_metrics['accuracy']}  (leaky, validation only)")
    print(f"  GB  HONEST F1-macro: {gb_metrics['f1_macro']}")
    print(f"  GB  HONEST AUC-ROC : {gb_metrics['auc_roc']}")
    print(f"  GB  HONEST 5-CV F1 : {gb_metrics['cv_f1_mean']} ± {gb_metrics['cv_f1_std']}")

    return gb, encoders, le


if __name__ == "__main__":
    DATA_PATH = os.path.join(ROOT, "data", "bengaluru_traffic_events.csv")
    MODEL_DIR = os.path.join(ROOT, "models")
    ASSETS_DIR = os.path.join(ROOT, "assets")
    train_pipeline(DATA_PATH, MODEL_DIR, ASSETS_DIR)
