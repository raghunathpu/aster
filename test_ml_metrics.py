"""
ASTER — ML Model Testing & Validation Script
============================================
Programmatically loads all trained models (Random Forest, Gradient Boosting, LightGBM Cascade),
re-creates the Stratified 80/20 Test Sets, runs inference, and verifies their metrics.
Acts as a unit test for model health and accuracy bounds.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.modeling.train import load_and_prepare
from src.modeling.train_lgb import prepare_lgb_features
from src.modeling.lgb_inference import FEATURE_ORDER

MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "bengaluru_traffic_events.csv")

def test_primary_models():
    print("\n" + "=" * 60)
    print("  Testing Primary Models (Triage Classifier)")
    print("=" * 60)
    
    if not os.path.exists(os.path.join(MODELS_DIR, "gb_main.pkl")):
        print("[ERROR] Primary model files not found. Run train.py first.")
        return False

    # Load models
    rf = joblib.load(os.path.join(MODELS_DIR, "rf_baseline.pkl"))
    gb = joblib.load(os.path.join(MODELS_DIR, "gb_main.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    
    # Prepare test set
    X, y_enc, _, _, _ = load_and_prepare(DATA_PATH)
    _, X_test, _, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    print(f"Test Set Size: {X_test.shape[0]} samples")

    # Evaluate RF
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average="macro")
    
    # Evaluate GB
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_f1 = f1_score(y_test, gb_pred, average="macro")

    print(f"\n[Random Forest Baseline]")
    print(f"  Accuracy:   {rf_acc*100:.2f}%")
    print(f"  F1 (macro): {rf_f1*100:.2f}%")
    
    print(f"\n[Gradient Boosting Main]")
    print(f"  Accuracy:   {gb_acc*100:.2f}%")
    print(f"  F1 (macro): {gb_f1*100:.2f}%")

    # Automated Assertions
    assert gb_acc > 0.85, f"GB Accuracy ({gb_acc:.4f}) is below the required 85% threshold."
    assert gb_f1 > 0.80, f"GB F1-Score ({gb_f1:.4f}) is below the required 80% threshold."
    print("\n[PASS] Primary Model tests PASSED (realistic metrics, target leakage resolved).")
    return True

def test_cascading_lgb_models():
    print("\n" + "=" * 60)
    print("  Testing Secondary Models (Cascading LightGBM)")
    print("=" * 60)

    lgb_dir = os.path.join(MODELS_DIR, "lgb")
    if not os.path.exists(os.path.join(lgb_dir, "road_closure_lgb.pkl")):
        print("[ERROR] LightGBM model files not found. Run train.py first.")
        return False

    # Load LGB models
    from src.modeling.lightgbm_model import LightGBMModel
    m_closure = LightGBMModel.load(os.path.join(lgb_dir, "road_closure_lgb.pkl"))
    m_priority = LightGBMModel.load(os.path.join(lgb_dir, "priority_lgb.pkl"))
    m_eis = LightGBMModel.load(os.path.join(lgb_dir, "eis_lgb.pkl"))
    m_res = LightGBMModel.load(os.path.join(lgb_dir, "resolution_lgb.pkl"))

    # Load data
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    X_full, df_processed = prepare_lgb_features(df_raw)

    # 1. Road Closure
    cl_features = [feat for feat in FEATURE_ORDER if feat != "requires_road_closure" and feat != "night_x_closure"]
    y_closure = df_processed["requires_road_closure"].astype(int)
    X_closure = X_full[cl_features]
    _, X_test_cl, _, y_test_cl = train_test_split(
        X_closure, y_closure, test_size=0.2, stratify=y_closure, random_state=42
    )
    cl_pred = m_closure.predict(X_test_cl)
    cl_f1 = f1_score(y_test_cl, cl_pred, average="macro")
    print(f"[Model 1: Road Closure Classifier]")
    print(f"  F1-Score (macro): {cl_f1*100:.2f}%")
    assert cl_f1 > 0.85, "Road closure classifier F1 is too low!"

    # 2. Priority
    y_priority = df_processed["priority_encoded"].astype(int)
    X_priority = X_full[FEATURE_ORDER]
    _, X_test_pr, _, y_test_pr = train_test_split(
        X_priority, y_priority, test_size=0.2, stratify=y_priority, random_state=42
    )
    pr_pred = m_priority.predict(X_test_pr)
    pr_f1 = f1_score(y_test_pr, pr_pred, average="macro")
    print(f"\n[Model 2: Priority Classifier]")
    print(f"  F1-Score (macro): {pr_f1*100:.2f}%")
    assert pr_f1 > 0.85, "Priority classifier F1 is too low!"

    # 3. EIS
    y_eis = df_processed["eis_scaled"]
    X_eis = X_full[cl_features]
    _, X_test_eis, _, y_test_eis = train_test_split(
        X_eis, y_eis, test_size=0.2, random_state=42
    )
    eis_pred = m_eis.predict(X_test_eis)
    eis_r2 = r2_score(y_test_eis, eis_pred)
    print(f"\n[Model 3: Event Impact Score (EIS) Regressor]")
    print(f"  R-squared (R²):   {eis_r2:.4f}")
    assert eis_r2 > 0.85, "EIS regressor R2 is too low!"

    # 4. Resolution Time
    valid_dur = df_processed[df_processed["duration_minutes"] > 0].copy()
    y_res = np.log1p(valid_dur["duration_minutes"])
    X_res = X_full.loc[valid_dur.index, FEATURE_ORDER]
    _, X_test_res, _, y_test_res = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )
    res_pred = m_res.predict(X_test_res)
    res_r2 = r2_score(y_test_res, res_pred)
    res_mae_orig = mean_absolute_error(np.expm1(y_test_res), np.expm1(res_pred))
    print(f"\n[Model 4: Resolution Time Regressor]")
    print(f"  Log-scale R²:     {res_r2:.4f}")
    print(f"  Original MAE:     {res_mae_orig:.2f} minutes")
    assert res_r2 > 0.45, "Resolution regressor R2 is too low!"

    print("\n[PASS] Cascading LightGBM Model tests PASSED.")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("  ASTER ML Test Suite — Model Metrics Validation")
    print("=" * 60)
    
    primary_ok = test_primary_models()
    lgb_ok = test_cascading_lgb_models()
    
    print("\n" + "=" * 60)
    if primary_ok and lgb_ok:
        print("  SUCCESS: ALL TESTS PASSED! Models are verified and healthy.")
    else:
        print("  FAILURE: SOME TESTS FAILED. Check the logs above.")
    print("=" * 60)
