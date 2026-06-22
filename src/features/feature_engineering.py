"""
ASTER – Feature Engineering
============================
Transforms the preprocessed DataFrame into a model-ready feature matrix.
Every feature is operationally grounded in Bengaluru traffic realities.
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# Peak-hour windows (IST)
# ──────────────────────────────────────────────
def _time_period(hour):
    if pd.isna(hour):
        return "unknown"
    h = int(hour)
    if 7 <= h <= 10:
        return "morning_peak"
    elif 17 <= h <= 21:
        return "evening_peak"
    elif 0 <= h <= 5:
        return "night"
    else:
        return "offpeak"


# ──────────────────────────────────────────────
# Season / quarter (monsoon is operationally distinct)
# ──────────────────────────────────────────────
def _season(month):
    if pd.isna(month):
        return "unknown"
    m = int(month)
    if m in [6, 7, 8, 9]:
        return "monsoon"
    elif m in [10, 11, 12]:
        return "post_monsoon"
    elif m in [1, 2]:
        return "winter"
    else:
        return "summer"


# ──────────────────────────────────────────────
# Spatial grid cell (500 m resolution approx)
# ──────────────────────────────────────────────
def _grid_cell(lat, lon, resolution=0.005):
    if pd.isna(lat) or pd.isna(lon):
        return "unknown"
    cell_lat = round(lat / resolution) * resolution
    cell_lon = round(lon / resolution) * resolution
    return f"{cell_lat:.3f}_{cell_lon:.3f}"


# ──────────────────────────────────────────────
# Historical frequency features
# ──────────────────────────────────────────────
def compute_historical_frequencies(df: pd.DataFrame, freq_maps: dict = None) -> tuple:
    """
    Add location-level and cause-level historical frequency columns.
    These capture how 'hotspot' a given junction/police station/cause is.
    If freq_maps is provided, uses those maps (inference mode). Otherwise, computes them (training mode).
    """
    if freq_maps is None:
        freq_maps = {}
        freq_maps["ps_freq"] = df["police_station"].value_counts().to_dict()
        if "junction" in df.columns:
            freq_maps["jn_freq"] = df["junction"].fillna("unknown").value_counts().to_dict()
        else:
            freq_maps["jn_freq"] = {}
        freq_maps["corr_freq"] = df["corridor"].value_counts().to_dict()
        
        freq_maps["cause_risk"] = (
            df.groupby("event_cause")["priority"]
            .apply(lambda x: (x == "High").mean())
            .to_dict()
        )
        freq_maps["corr_closure"] = (
            df.groupby("corridor")["road_closure_flag"]
            .mean()
            .to_dict()
        )

    # Police station event count
    df["ps_event_freq"] = df["police_station"].map(freq_maps["ps_freq"]).fillna(1)

    # Junction event count
    if "junction" in df.columns:
        df["junction_event_freq"] = df["junction"].fillna("unknown").map(freq_maps.get("jn_freq", {})).fillna(1)
    else:
        df["junction_event_freq"] = 1

    # Corridor event count
    df["corridor_event_freq"] = df["corridor"].map(freq_maps["corr_freq"]).fillna(1)

    # Event cause risk rate (fraction that are High priority)
    df["cause_risk_rate"] = df["event_cause"].map(freq_maps["cause_risk"]).fillna(0.5)

    # Corridor road-closure rate
    df["corridor_closure_rate"] = df["corridor"].map(freq_maps["corr_closure"]).fillna(0.0)

    return df, freq_maps


# ──────────────────────────────────────────────
# Main feature builder
# ──────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "event_type", "event_cause",
    "corridor", "zone", "veh_type",
    "time_period", "season", "day_type",
]

NUMERIC_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "ps_event_freq", "junction_event_freq",
    "corridor_event_freq", "cause_risk_rate",
    "corridor_closure_rate", "lat_norm", "lon_norm",
]


def build_features(df: pd.DataFrame, freq_maps: dict = None) -> tuple:
    """
    Add all engineered features to the DataFrame.
    Returns the modified DataFrame and the fitted frequency maps.
    """
    # ── Time features ──────────────────────────
    if "start_local" in df.columns:
        df["hour"] = df["start_local"].dt.hour.fillna(-1).astype(int)
        df["day_of_week"] = df["start_local"].dt.dayofweek.fillna(-1).astype(int)  # 0=Mon
        df["month"] = df["start_local"].dt.month.fillna(-1).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["time_period"] = df["hour"].map(_time_period)
        df["season"] = df["month"].map(_season)
        df["day_type"] = df["is_weekend"].map({0: "weekday", 1: "weekend"})
    else:
        for col in ["hour", "day_of_week", "month", "is_weekend"]:
            df[col] = 0
        df["time_period"] = "unknown"
        df["season"] = "unknown"
        df["day_type"] = "weekday"

    # ── Spatial features ──────────────────────
    df["lat_norm"] = (df["latitude"] - 12.97) / 0.2   # Bengaluru centroid
    df["lon_norm"] = (df["longitude"] - 77.59) / 0.2
    df["grid_cell"] = df.apply(
        lambda r: _grid_cell(r["latitude"], r["longitude"]), axis=1
    )

    # ── Fill remaining categoricals ───────────
    df["veh_type"] = df["veh_type"].fillna("unknown")
    df["zone"] = df["zone"].fillna("Unknown")

    # ── Historical frequency features ─────────
    df, freq_maps = compute_historical_frequencies(df, freq_maps)

    return df, freq_maps


def encode_for_model(df: pd.DataFrame, fit_encoders: dict = None):
    """
    One-hot encode categoricals, return (X, encoder_map).
    If fit_encoders is provided, uses those categories (inference mode).
    """
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    if fit_encoders is None:
        # Training mode: learn categories
        encoders = {}
        dummies_list = []
        for col in cat_cols:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=False)
            encoders[col] = list(dummies.columns)
            dummies_list.append(dummies)
        X_cat = pd.concat(dummies_list, axis=1)
    else:
        # Inference mode: align to trained columns
        encoders = fit_encoders
        dummies_list = []
        for col in cat_cols:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=False)
            for c in encoders[col]:
                if c not in dummies.columns:
                    dummies[c] = 0
            dummies = dummies[encoders[col]]
            dummies_list.append(dummies)
        X_cat = pd.concat(dummies_list, axis=1)

    X_num = df[num_cols].fillna(0)
    X = pd.concat([X_num, X_cat], axis=1)
    return X, encoders


def get_feature_names(encoders: dict) -> list:
    """Return flat list of all feature names after encoding."""
    num_names = NUMERIC_FEATURES[:]
    cat_names = []
    for col_dummies in encoders.values():
        cat_names.extend(col_dummies)
    return num_names + cat_names
