"""
ASTER – Inference Engine
=========================
Loads trained model artefacts and provides prediction interface
for both single events and batch inputs.
"""

import os, sys, joblib
import numpy as np
import pandas as pd
import zoneinfo

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.features.feature_engineering import build_features, encode_for_model

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

NAMED_CORRIDORS = {
    "Mysore Road", "Bellary Road 1", "Bellary Road 2",
    "Tumkur Road", "Hosur Road", "ORR North 1", "Old Madras Road",
    "Magadi Road", "ORR East 1", "ORR North 2", "Bannerghata Road",
    "ORR East 2", "West of Chord Road", "ORR West 1", "CBD 2",
    "Hennur Main Road", "IRR(Thanisandra road)", "Varthur Road", "Old Airport Road",
}

HIGH_IMPACT_CAUSES = {
    "accident", "construction", "public_event",
    "protest", "procession", "vip_movement", "water_logging",
}


class ASTERPredictor:
    """
    Wraps trained ASTER model for clean prediction calls.

    Usage:
        predictor = ASTERPredictor(model_dir="models/")
        result    = predictor.predict_single(event_dict)
    """

    LABEL_ORDER = ["Low", "Medium", "High"]

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._load_artefacts()

    def _load_artefacts(self):
        self.model    = joblib.load(os.path.join(self.model_dir, "gb_main.pkl"))
        self.encoders = joblib.load(os.path.join(self.model_dir, "encoders.pkl"))
        self.freq_maps = joblib.load(os.path.join(self.model_dir, "freq_maps.pkl"))
        self.le       = joblib.load(os.path.join(self.model_dir, "label_encoder.pkl"))
        self.feature_names = joblib.load(os.path.join(self.model_dir, "feature_names.pkl"))
        fi_path = os.path.join(self.model_dir, "feature_importance.csv")
        self.feature_importance = pd.read_csv(fi_path) if os.path.exists(fi_path) else None

    def _prepare_single(self, event: dict) -> pd.DataFrame:
        """Convert a raw event dict to a one-row feature DataFrame."""
        row = pd.DataFrame([event])

        defaults = {
            "event_type":          "unplanned",
            "event_cause":         "others",
            "latitude":            12.97,
            "longitude":           77.59,
            "requires_road_closure": False,
            "priority":            "High",
            "corridor":            "Non-corridor",
            "zone":                "Unknown",
            "veh_type":            "unknown",
            "police_station":      "Unknown",
            "junction":            None,
            "start_datetime":      pd.Timestamp.now(tz="UTC"),
        }
        for k, v in defaults.items():
            if k not in row.columns:
                row[k] = v

        row["start_datetime"] = pd.to_datetime(row["start_datetime"], utc=True, errors="coerce")
        row["start_local"]    = row["start_datetime"].dt.tz_convert(IST)

        row["corridor"]         = row["corridor"].fillna("Non-corridor")
        row["is_named_corridor"] = row["corridor"].isin(NAMED_CORRIDORS).astype(int)
        row["road_closure_flag"] = row["requires_road_closure"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        ).astype(int)
        row["priority"]   = row["priority"].fillna("High")
        row["veh_type"]   = row["veh_type"].fillna("unknown")
        row["zone"]       = row["zone"].fillna("Unknown")
        row["police_station"] = row["police_station"].fillna("Unknown")

        row, _ = build_features(row, freq_maps=self.freq_maps)
        return row

    def predict_single(self, event: dict) -> dict:
        row  = self._prepare_single(event)
        X, _ = encode_for_model(row, fit_encoders=self.encoders)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        probs     = self.model.predict_proba(X)[0]
        pred_idx  = int(np.argmax(probs))
        pred_label = self.le.classes_[pred_idx]
        prob_dict  = {self.le.classes_[i]: round(float(p), 4) for i, p in enumerate(probs)}

        return {
            "predicted_tier":  pred_label,
            "probabilities":   prob_dict,
            "confidence":      round(float(probs[pred_idx]), 4),
            "impact_score_raw": int(self._raw_score(event)),
        }

    def _raw_score(self, event: dict) -> int:
        score = 1
        if str(event.get("requires_road_closure", False)).lower() in ["true", "1", "yes"]:
            score += 2
        if event.get("priority") == "High":
            score += 1
        if event.get("event_cause") in HIGH_IMPACT_CAUSES:
            score += 1
        if event.get("corridor") in NAMED_CORRIDORS:
            score += 1
        return min(score, 6)

    def get_top_factors(self, event: dict, top_n: int = 5) -> list:
        if self.feature_importance is None:
            return []
        row  = self._prepare_single(event)
        X, _ = encode_for_model(row, fit_encoders=self.encoders)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        active = X.columns[(X.values[0] != 0)].tolist()
        fi = self.feature_importance.copy()
        fi["active"] = fi["feature"].isin(active)
        fi = fi.sort_values(["active", "importance"], ascending=[False, False])
        return [_clean_feature_name(f) for f in fi.head(top_n)["feature"].tolist()]


def _clean_feature_name(name: str) -> str:
    clean_map = {
        "road_closure_flag":     "Road closure required",
        "is_named_corridor":     "Named high-traffic corridor",
        "cause_risk_rate":       "Historical risk rate for this cause",
        "corridor_closure_rate": "Corridor closure history",
        "ps_event_freq":         "Police station incident volume",
        "junction_event_freq":   "Junction hotspot frequency",
        "corridor_event_freq":   "Corridor event frequency",
        "hour":                  "Hour of day",
        "day_of_week":           "Day of week",
        "is_weekend":            "Weekend flag",
        "lat_norm":              "Location (latitude)",
        "lon_norm":              "Location (longitude)",
        "month":                 "Month of year",
    }
    if name in clean_map:
        return clean_map[name]
    for prefix, label in [
        ("event_cause_", "Event cause: "), ("event_type_", "Event type: "),
        ("priority_",    "Priority: "),    ("corridor_",   "Corridor: "),
        ("zone_",        "Zone: "),        ("veh_type_",   "Vehicle type: "),
        ("time_period_", "Time period: "),
    ]:
        if name.startswith(prefix):
            return label + name[len(prefix):].replace("_", " ").title()
    return name.replace("_", " ").title()
