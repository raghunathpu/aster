"""
ASTER — LightGBM Cascading Model Trainer
========================================
Trains the 4 cascading LightGBM models matching ASTER's inference pipeline:
  1. Road Closure Classifier (predicts requires_road_closure)
  2. Priority Classifier (predicts priority_encoded using road closure)
  3. Event Impact Score Regressor (predicts impact score scaled 0-1)
  4. Resolution Time Regressor (predicts log-transformed duration in minutes)

Saves models to models/lgb/ and saves validation metrics to models/lgb/evaluation_metrics.json.
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
import geohash2 as gh
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.modeling.lightgbm_model import LightGBMModel
from src.modeling.lgb_inference import EVENT_CAUSES, VEHICLE_TYPES, EVENT_CAUSE_SEVERITY_WEIGHTS, FEATURE_ORDER

def prepare_lgb_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and build the exact feature matrix for LightGBM models."""
    df = df_raw.copy()

    # Time parse
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    # Convert to IST timezone
    IST = "Asia/Kolkata"
    df["start_local"] = df["start_datetime"].dt.tz_convert(IST)
    
    # Fill null coordinates
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] > 1) & (df["longitude"] > 1)].copy()

    # Standardize values
    df["event_cause"] = df["event_cause"].fillna("others").astype(str).str.strip().str.lower().str.replace(" ", "_")
    df["veh_type"] = df["veh_type"].fillna("unknown").astype(str).str.strip().str.lower().str.replace(" ", "_")
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df["zone"] = df["zone"].fillna("Unknown").astype(str).str.strip()
    df["event_type"] = df["event_type"].fillna("unplanned").astype(str).str.strip().str.lower()
    df["requires_road_closure"] = df["requires_road_closure"].astype(str).str.lower().isin(["true", "1", "yes", "1.0"]).astype(float)
    df["priority_encoded"] = df["priority"].astype(str).str.lower().isin(["high"]).astype(float)

    # Compute duration
    df["closed_datetime"] = pd.to_datetime(df["closed_datetime"], utc=True, errors="coerce")
    delta = (df["closed_datetime"] - df["start_datetime"]).dt.total_seconds()
    df["duration_minutes"] = np.where(delta > 0, delta / 60.0, np.nan)

    # Compute impact score (base logic from data_loader)
    score = pd.Series(1.0, index=df.index)
    score += df["requires_road_closure"] * 2.0
    score += df["priority_encoded"]
    high_causes = {"accident", "construction", "public_event", "protest", "procession", "vip_movement", "water_logging"}
    score += df["event_cause"].isin(high_causes).astype(float)
    named_corridors = {
        "Mysore Road", "Bellary Road 1", "Bellary Road 2", "Tumkur Road", "Hosur Road",
        "ORR North 1", "Old Madras Road", "Magadi Road", "ORR East 1", "ORR North 2",
        "Bannerghata Road", "ORR East 2", "West of Chord Road", "ORR West 1", "CBD 2",
        "Hennur Main Road", "IRR(Thanisandra road)", "Varthur Road", "Old Airport Road"
    }
    score += df["corridor"].isin(named_corridors).astype(float)
    df["impact_score"] = score.clip(1, 6)
    df["eis_scaled"] = df["impact_score"] / 6.0

    # Build historical tables
    df["geohash_key"] = df.apply(
        lambda r: gh.encode(float(r["latitude"]), float(r["longitude"]), precision=6), axis=1
    )

    # Group metrics for geohash
    gh_stats = {}
    for g_key, group in df.groupby("geohash_key"):
        total = len(group)
        closures = sum(group["requires_road_closure"])
        high_prios = sum(df.loc[group.index, "priority_encoded"])
        avg_eis = group["eis_scaled"].mean()
        gh_stats[g_key] = {
            "total": total,
            "closure_rate": closures / total,
            "high_prio_rate": high_prios / total,
            "avg_eis": avg_eis
        }

    # Group metrics for corridor
    corr_stats = {}
    for c_key, group in df.groupby("corridor"):
        total = len(group)
        closures = sum(group["requires_road_closure"])
        avg_eis = group["eis_scaled"].mean()
        corr_stats[c_key] = {
            "total": total,
            "closure_rate": closures / total,
            "avg_eis": avg_eis
        }

    # Build feature columns
    features = []
    
    for idx, row in df.iterrows():
        dt = row["start_local"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cause = row["event_cause"]
        veh = row["veh_type"]
        corridor = row["corridor"]
        zone = row["zone"]
        geohash_str = row["geohash_key"]
        
        f = {}
        
        # Temporal
        f["hour"] = float(dt.hour)
        f["hour_sin"] = math.sin(2 * math.pi * dt.hour / 24)
        f["hour_cos"] = math.cos(2 * math.pi * dt.hour / 24)
        f["day_of_week"] = float(dt.weekday())
        f["dow_sin"] = math.sin(2 * math.pi * dt.weekday() / 7)
        f["dow_cos"] = math.cos(2 * math.pi * dt.weekday() / 7)
        f["month"] = float(dt.month)
        f["month_sin"] = math.sin(2 * math.pi * dt.month / 12)
        f["month_cos"] = math.cos(2 * math.pi * dt.month / 12)
        f["is_weekend"] = 1.0 if dt.weekday() >= 5 else 0.0
        f["is_rush_hour"] = 1.0 if dt.hour in (8, 9, 10, 17, 18, 19) else 0.0
        f["is_night"] = 1.0 if dt.hour < 6 or dt.hour >= 22 else 0.0
        f["time_period"] = (
            0.0 if dt.hour < 6 else
            1.0 if dt.hour < 12 else
            2.0 if dt.hour < 17 else
            3.0 if dt.hour < 21 else
            4.0
        )

        # Causes
        for c in EVENT_CAUSES:
            f[f"cause_{c}"] = 1.0 if cause == c else 0.0
        f["cause_severity_weight"] = EVENT_CAUSE_SEVERITY_WEIGHTS.get(cause, 0.3)

        # Event type
        f["is_planned"] = 1.0 if row["event_type"] == "planned" else 0.0

        # Vehicles
        for v in VEHICLE_TYPES:
            f[f"veh_{v}"] = 1.0 if veh == v else 0.0
        f["veh_unknown"] = 1.0 if veh in ("unknown", "") else 0.0
        f["has_vehicle_info"] = 0.0 if veh in ("unknown", "") else 1.0

        # Requires road closure (will be actual value in base row)
        f["requires_road_closure"] = float(row["requires_road_closure"])

        # Corridor
        f["is_corridor"] = 0.0 if corridor == "Non-corridor" else 1.0
        top_corridors = [
            "Mysore Road", "Bellary Road 1", "Tumkur Road", "Bellary Road 2",
            "Hosur Road", "ORR North 1", "Old Madras Road", "Magadi Road",
        ]
        for tc in top_corridors:
            tc_clean = tc.replace(" ", "_").lower()
            f[f"corridor_{tc_clean}"] = 1.0 if corridor == tc else 0.0
        f["is_orr"] = 1.0 if "ORR" in corridor else 0.0
        f["is_cbd"] = 1.0 if "CBD" in corridor else 0.0

        # Zone
        f["has_zone_info"] = 0.0 if zone == "Unknown" else 1.0
        zone_groups = {"Central": 0, "North": 0, "South": 0, "East": 0, "West": 0}
        for zg in zone_groups:
            f[f"zone_{zg.lower()}"] = 1.0 if zg in zone else 0.0

        # Geospatial
        f["latitude"] = lat
        f["longitude"] = lon
        center_lat, center_lon = 12.9716, 77.5946
        f["dist_from_center_km"] = math.sqrt(
            ((lat - center_lat) * 111.0) ** 2 + ((lon - center_lon) * 85.0) ** 2
        )
        
        # End coordinates
        end_lat = row.get("endlatitude")
        end_lon = row.get("endlongitude")
        if end_lat is not None and end_lon is not None and not pd.isna(end_lat) and not pd.isna(end_lon) and end_lat > 1:
            f["has_end_coords"] = 1.0
            # Rough distance calculation
            f["spatial_extent_km"] = math.sqrt(
                ((lat - end_lat) * 111.0) ** 2 + ((lon - end_lon) * 85.0) ** 2
            )
        else:
            f["has_end_coords"] = 0.0
            f["spatial_extent_km"] = 0.0

        # Lookups
        gh_s = gh_stats.get(geohash_str, {})
        corr_s = corr_stats.get(corridor, {})

        f["corridor_event_count"] = float(corr_s.get("total", 10.0))
        f["junction_event_count"] = float(gh_s.get("total", 5.0))
        f["cause_at_location_count"] = float(gh_s.get("total", 5.0)) * 0.2
        f["is_repeat_location"] = 1.0 if f["junction_event_count"] > 1 else 0.0

        f["geohash_total_events"] = float(gh_s.get("total", 1.0))
        f["geohash_closure_rate"] = float(gh_s.get("closure_rate", 0.1))
        f["geohash_high_priority_rate"] = float(gh_s.get("high_prio_rate", 0.2))
        f["geohash_avg_eis"] = float(gh_s.get("avg_eis", 0.3))

        f["corridor_closure_rate"] = float(corr_s.get("closure_rate", 0.1))
        f["corridor_avg_eis"] = float(corr_s.get("avg_eis", 0.3))

        # Interactions
        f["rush_hour_x_planned"] = f["is_rush_hour"] * f["is_planned"]
        f["weekend_x_planned"] = f["is_weekend"] * f["is_planned"]
        f["night_x_closure"] = f["is_night"] * f["requires_road_closure"]
        f["corridor_x_rush"] = f["is_corridor"] * f["is_rush_hour"]
        f["severity_x_corridor"] = f["cause_severity_weight"] * f["is_corridor"]
        f["center_dist_x_severity"] = f["dist_from_center_km"] * f["cause_severity_weight"]
        
        # Add index back for targets alignment
        f["_index"] = idx
        features.append(f)

    df_feats = pd.DataFrame(features)
    df_feats.index = df_feats["_index"]
    df_feats = df_feats.drop(columns=["_index"])
    
    # Align rows with df
    df = df.loc[df_feats.index]
    return df_feats, df

def train_lgb_pipeline(data_path: str, output_dir: str):
    """Run full LightGBM training and evaluation pipeline."""
    print("Loading raw events for LightGBM training...")
    df_raw = pd.read_csv(data_path, low_memory=False)
    
    print("Building features for LightGBM...")
    X_full, df_processed = prepare_lgb_features(df_raw)
    print(f"Constructed LightGBM feature matrix: {X_full.shape}")

    # Create target directories
    lgb_dir = os.path.join(output_dir, "lgb")
    os.makedirs(lgb_dir, exist_ok=True)

    metrics = {}

    # -------------------------------------------------------------
    # 1. Road Closure Model
    # Target: requires_road_closure (binary classification)
    # Features: exclude requires_road_closure and night_x_closure
    # -------------------------------------------------------------
    print("\nTraining Model 1/4: Road Closure Classifier...")
    cl_features = [feat for feat in FEATURE_ORDER if feat != "requires_road_closure" and feat != "night_x_closure"]
    
    # Target values
    y_closure = df_processed["requires_road_closure"].astype(int)
    X_closure = X_full[cl_features].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_closure, y_closure, test_size=0.2, stratify=y_closure, random_state=42
    )

    # Train
    model_closure = LightGBMModel(
        model_type="classification",
        params={
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_samples": 15,
            "subsample": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1
        }
    )
    model_closure.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model_closure.predict(X_test)
    y_prob = model_closure.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro")
    f1_w = f1_score(y_test, y_pred, average="weighted")
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  F1 (macro): {f1_m:.4f}")
    print(f"  ROC-AUC:    {roc:.4f}")
    
    metrics["road_closure"] = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "roc_auc": round(roc, 4)
    }
    
    model_closure.save(os.path.join(lgb_dir, "road_closure_lgb.pkl"))

    # -------------------------------------------------------------
    # 2. Priority Model
    # Target: priority_encoded (binary classification)
    # Features: Include requires_road_closure and night_x_closure
    # -------------------------------------------------------------
    print("\nTraining Model 2/4: Priority Classifier...")
    y_priority = df_processed["priority_encoded"].astype(int)
    X_priority = X_full[FEATURE_ORDER].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_priority, y_priority, test_size=0.2, stratify=y_priority, random_state=42
    )

    # Train
    model_priority = LightGBMModel(
        model_type="classification",
        params={
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_samples": 15,
            "subsample": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1
        }
    )
    model_priority.fit(X_train, y_train)

    # Evaluate
    y_pred = model_priority.predict(X_test)
    y_prob = model_priority.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro")
    f1_w = f1_score(y_test, y_pred, average="weighted")
    roc = roc_auc_score(y_test, y_prob)

    print(f"  Accuracy:   {acc:.4f}")
    print(f"  F1 (macro): {f1_m:.4f}")
    print(f"  ROC-AUC:    {roc:.4f}")

    metrics["priority"] = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "roc_auc": round(roc, 4)
    }

    model_priority.save(os.path.join(lgb_dir, "priority_lgb.pkl"))

    # -------------------------------------------------------------
    # 3. Event Impact Score (EIS) Model
    # Target: eis_scaled (regression, range [0, 1])
    # Features: exclude requires_road_closure and night_x_closure
    # -------------------------------------------------------------
    print("\nTraining Model 3/4: Event Impact Score (EIS) Regressor...")
    y_eis = df_processed["eis_scaled"]
    X_eis = X_full[cl_features].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_eis, y_eis, test_size=0.2, random_state=42
    )

    # Train
    model_eis = LightGBMModel(
        model_type="regression",
        params={
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_samples": 15,
            "subsample": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1
        }
    )
    model_eis.fit(X_train, y_train)

    # Evaluate
    y_pred = model_eis.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2: {r2:.4f}")

    metrics["eis"] = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }

    model_eis.save(os.path.join(lgb_dir, "eis_lgb.pkl"))

    # -------------------------------------------------------------
    # 4. Resolution Time Model
    # Target: np.log1p(duration_minutes) (regression, only valid durations)
    # Features: all features in FEATURE_ORDER
    # -------------------------------------------------------------
    print("\nTraining Model 4/4: Resolution Time Regressor...")
    valid_dur = df_processed[df_processed["duration_minutes"] > 0].copy()
    y_res = np.log1p(valid_dur["duration_minutes"])
    X_res = X_full.loc[valid_dur.index, FEATURE_ORDER].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # Train
    model_res = LightGBMModel(
        model_type="regression",
        params={
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_samples": 10,
            "subsample": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1
        }
    )
    model_res.fit(X_train, y_train)

    # Evaluate
    y_pred = model_res.predict(X_test)
    
    # Metrics on log scale
    mae_log = mean_absolute_error(y_test, y_pred)
    rmse_log = math.sqrt(mean_squared_error(y_test, y_pred))
    r2_res = r2_score(y_test, y_pred)

    # Metrics on original scale (minutes)
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse_orig = math.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

    print(f"  Log-scale MAE:  {mae_log:.4f}")
    print(f"  Log-scale R2:   {r2_res:.4f}")
    print(f"  Original MAE:   {mae_orig:.2f} mins")
    print(f"  Original RMSE:  {rmse_orig:.2f} mins")

    metrics["resolution"] = {
        "log_mae": round(mae_log, 4),
        "log_rmse": round(rmse_log, 4),
        "r2": round(r2_res, 4),
        "original_mae_min": round(mae_orig, 2),
        "original_rmse_min": round(rmse_orig, 2)
    }

    model_res.save(os.path.join(lgb_dir, "resolution_lgb.pkl"))

    # Save metrics JSON
    metrics_path = os.path.join(lgb_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OK] Trained all cascading LightGBM models. Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    DATA_PATH = os.path.join(ROOT, "data", "bengaluru_traffic_events.csv")
    OUTPUT_DIR = os.path.join(ROOT, "models")
    train_lgb_pipeline(DATA_PATH, OUTPUT_DIR)
