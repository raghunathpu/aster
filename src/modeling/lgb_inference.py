"""
ASTER — LightGBM Inference Service
===================================
Loads saved LightGBM models from Gridvision, builds feature vectors,
and computes cascading predictions for new events.
"""
import os
import sys
import types
import numpy as np
import pandas as pd
import geohash2 as gh
import math
from pathlib import Path

# Add local path and patch sys.modules for pickle deserialization
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.modeling.lightgbm_model import LightGBMModel

# Register sys modules for pickle to load gridvision classes correctly
if 'ml' not in sys.modules:
    ml_mod = types.ModuleType('ml')
    sys.modules['ml'] = ml_mod
if 'ml.models' not in sys.modules:
    ml_models_mod = types.ModuleType('ml.models')
    sys.modules['ml.models'] = ml_models_mod
if 'ml.models.lightgbm_model' not in sys.modules:
    ml_models_lgb_mod = types.ModuleType('ml.models.lightgbm_model')
    ml_models_lgb_mod.LightGBMModel = LightGBMModel
    sys.modules['ml.models.lightgbm_model'] = ml_models_lgb_mod

EVENT_CAUSES = [
    "vehicle_breakdown", "accident", "construction", "pot_holes",
    "water_logging", "tree_fall", "road_conditions", "congestion",
    "public_event", "procession", "vip_movement", "protest",
    "others", "debris", "fog_low_visibility", "test_demo",
]

VEHICLE_TYPES = [
    "bmtc_bus", "ksrtc_bus", "private_bus", "heavy_vehicle",
    "truck", "lcv", "private_car", "taxi", "auto", "others",
]

EVENT_CAUSE_SEVERITY_WEIGHTS = {
    "accident": 0.90,
    "protest": 0.85,
    "procession": 0.80,
    "vip_movement": 0.75,
    "public_event": 0.70,
    "tree_fall": 0.65,
    "water_logging": 0.60,
    "construction": 0.55,
    "vehicle_breakdown": 0.50,
    "congestion": 0.45,
    "road_conditions": 0.40,
    "pot_holes": 0.35,
    "debris": 0.30,
    "fog_low_visibility": 0.25,
    "others": 0.30,
    "test_demo": 0.05,
}

FEATURE_ORDER = [
    'hour', 'hour_sin', 'hour_cos', 'day_of_week', 'dow_sin', 'dow_cos',
    'month', 'month_sin', 'month_cos', 'is_weekend', 'is_rush_hour',
    'is_night', 'time_period', 'cause_vehicle_breakdown', 'cause_accident',
    'cause_construction', 'cause_pot_holes', 'cause_water_logging',
    'cause_tree_fall', 'cause_road_conditions', 'cause_congestion',
    'cause_public_event', 'cause_procession', 'cause_vip_movement',
    'cause_protest', 'cause_others', 'cause_debris', 'cause_fog_low_visibility',
    'cause_test_demo', 'cause_severity_weight', 'is_planned', 'veh_bmtc_bus',
    'veh_ksrtc_bus', 'veh_private_bus', 'veh_heavy_vehicle', 'veh_truck',
    'veh_lcv', 'veh_private_car', 'veh_taxi', 'veh_auto', 'veh_others',
    'veh_unknown', 'has_vehicle_info', 'requires_road_closure', 'is_corridor',
    'corridor_mysore_road', 'corridor_bellary_road_1', 'corridor_tumkur_road',
    'corridor_bellary_road_2', 'corridor_hosur_road', 'corridor_orr_north_1',
    'corridor_old_madras_road', 'corridor_magadi_road', 'is_orr', 'is_cbd',
    'has_zone_info', 'zone_central', 'zone_north', 'zone_south', 'zone_east',
    'zone_west', 'latitude', 'longitude', 'dist_from_center_km',
    'has_end_coords', 'spatial_extent_km', 'corridor_event_count',
    'junction_event_count', 'cause_at_location_count', 'is_repeat_location',
    'geohash_total_events', 'geohash_closure_rate', 'geohash_high_priority_rate',
    'geohash_avg_eis', 'corridor_closure_rate', 'corridor_avg_eis',
    'rush_hour_x_planned', 'weekend_x_planned', 'night_x_closure',
    'corridor_x_rush', 'severity_x_corridor', 'center_dist_x_severity'
]

class LGBInferenceService:
    def __init__(self, models_dir=None, dataset_path=None):
        if models_dir is None:
            models_dir = os.path.join(ROOT, "models", "lgb")
        if dataset_path is None:
            dataset_path = os.path.join(ROOT, "models", "processed_dataset.csv")

        # Load LightGBM model pickles
        try:
            self.model_closure = LightGBMModel.load(os.path.join(models_dir, "road_closure_lgb.pkl"))
            self.model_priority = LightGBMModel.load(os.path.join(models_dir, "priority_lgb.pkl"))
            self.model_eis = LightGBMModel.load(os.path.join(models_dir, "eis_lgb.pkl"))
            self.model_res = LightGBMModel.load(os.path.join(models_dir, "resolution_lgb.pkl"))
            print("Successfully loaded ASTER-LGB models.")
        except Exception as e:
            print(f"Error loading ASTER-LGB models: {e}")
            self.model_closure = None
            self.model_priority = None
            self.model_eis = None
            self.model_res = None

        # Build lookup tables for historical stats
        self.geohash_stats = {}
        self.corridor_stats = {}
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                self._build_stats(df)
                print(f"Built historical stats from {len(df)} records.")
            except Exception as e:
                print(f"Error building historical stats: {e}")

    def _build_stats(self, df: pd.DataFrame):
        # Clean columns if needed
        # We need to map lat/lon to geohash
        if "requires_road_closure" not in df.columns:
            df["requires_road_closure"] = df.get("road_closure_flag", 0)
        
        # Calculate geohash
        df["geohash_key"] = df.apply(
            lambda r: gh.encode(float(r["latitude"]), float(r["longitude"]), precision=6) 
            if pd.notnull(r["latitude"]) and pd.notnull(r["longitude"]) else "", axis=1
        )

        # Priority binary
        df["prio_bin"] = df["priority"].apply(lambda p: 1 if str(p).strip().lower() == "high" else 0)

        # Average EIS
        # Check if impact_score exists and scale to 0-1
        if "impact_score" in df.columns:
            # Let's say max score is 6, so scale it
            df["eis_scaled"] = df["impact_score"] / 6.0
        else:
            df["eis_scaled"] = 0.3

        # Group by Geohash
        gh_grouped = df.groupby("geohash_key")
        for g_key, group in gh_grouped:
            if g_key == "":
                continue
            total = len(group)
            closures = sum(group["requires_road_closure"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"]))
            high_prios = sum(group["prio_bin"])
            avg_eis = group["eis_scaled"].mean()
            self.geohash_stats[g_key] = {
                "total": total,
                "closure_rate": closures / total,
                "high_prio_rate": high_prios / total,
                "avg_eis": avg_eis
            }

        # Group by Corridor
        corr_grouped = df.groupby("corridor")
        for c_key, group in corr_grouped:
            total = len(group)
            closures = sum(group["requires_road_closure"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"]))
            avg_eis = group["eis_scaled"].mean()
            self.corridor_stats[c_key] = {
                "closure_rate": closures / total,
                "avg_eis": avg_eis
            }

    def predict(self, event_data: dict) -> dict:
        """
        Run cascading predictions.
        event_data contains:
          - start_datetime: pd.Timestamp or string
          - event_type: "planned" or "unplanned"
          - event_cause: str (e.g., "accident")
          - vehicle_type: str (e.g., "bmtc_bus")
          - latitude: float
          - longitude: float
          - corridor: str
          - zone: str
        """
        # If models didn't load, return fallback
        if not self.model_closure:
            return {
                "requires_road_closure": 0,
                "requires_road_closure_prob": 0.25,
                "priority_encoded": 0,
                "priority": "Low",
                "priority_prob": 0.15,
                "event_impact_score": 0.35,
                "resolution_time_min": 45.0
            }

        dt = pd.to_datetime(event_data.get("start_datetime", pd.Timestamp.now()))
        lat = float(event_data.get("latitude", 12.9716))
        lon = float(event_data.get("longitude", 77.5946))
        geohash_str = gh.encode(lat, lon, precision=6)
        
        cause = str(event_data.get("event_cause", "others")).strip().lower().replace(" ", "_")
        veh = str(event_data.get("vehicle_type", "unknown")).strip().lower().replace(" ", "_")
        corridor = event_data.get("corridor", "Non-corridor")
        zone = str(event_data.get("zone", "Unknown")).strip()

        # Build features dict
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
        f["is_planned"] = 1.0 if event_data.get("event_type", "unplanned") == "planned" else 0.0

        # Vehicles
        for v in VEHICLE_TYPES:
            f[f"veh_{v}"] = 1.0 if veh == v else 0.0
        f["veh_unknown"] = 1.0 if veh in ("unknown", "") else 0.0
        f["has_vehicle_info"] = 0.0 if veh in ("unknown", "") else 1.0

        # Placeholder for closure
        f["requires_road_closure"] = 0.0

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
        f["has_end_coords"] = 0.0
        f["spatial_extent_km"] = 0.0

        # Event frequency
        f["corridor_event_count"] = 1.0
        f["junction_event_count"] = 1.0
        f["cause_at_location_count"] = 1.0
        f["is_repeat_location"] = 1.0

        # Lookups
        gh_s = self.geohash_stats.get(geohash_str, {})
        f["geohash_total_events"] = float(gh_s.get("total", 1.0))
        f["geohash_closure_rate"] = float(gh_s.get("closure_rate", 0.1))
        f["geohash_high_priority_rate"] = float(gh_s.get("high_prio_rate", 0.2))
        f["geohash_avg_eis"] = float(gh_s.get("avg_eis", 0.3))

        corr_s = self.corridor_stats.get(corridor, {})
        f["corridor_closure_rate"] = float(corr_s.get("closure_rate", 0.1))
        f["corridor_avg_eis"] = float(corr_s.get("avg_eis", 0.3))

        # Interactions
        f["rush_hour_x_planned"] = f["is_rush_hour"] * f["is_planned"]
        f["weekend_x_planned"] = f["is_weekend"] * f["is_planned"]
        f["night_x_closure"] = f["is_night"] * f["requires_road_closure"]
        f["corridor_x_rush"] = f["is_corridor"] * f["is_rush_hour"]
        f["severity_x_corridor"] = f["cause_severity_weight"] * f["is_corridor"]
        f["center_dist_x_severity"] = f["dist_from_center_km"] * f["cause_severity_weight"]

        # Run predictions in cascade
        # 1. Road Closure
        cl_features = [feat for feat in FEATURE_ORDER if feat != "requires_road_closure"]
        cl_vector = pd.DataFrame([[f[feat] for feat in cl_features]], columns=cl_features)
        road_closure_prob = float(self.model_closure.predict_proba(cl_vector)[0, 1])
        road_closure_pred = int(self.model_closure.predict(cl_vector)[0])

        # Inject road closure pred back into features
        f["requires_road_closure"] = float(road_closure_pred)
        f["night_x_closure"] = f["is_night"] * f["requires_road_closure"]

        # 2. Priority
        feat_vector = pd.DataFrame([[f[feat] for feat in FEATURE_ORDER]], columns=FEATURE_ORDER)
        priority_prob = float(self.model_priority.predict_proba(feat_vector)[0, 1])
        priority_pred = int(self.model_priority.predict(feat_vector)[0])

        # 3. EIS (excludes requires_road_closure)
        eis_vector = pd.DataFrame([[f[feat] for feat in cl_features]], columns=cl_features)
        predicted_eis = float(self.model_eis.predict(eis_vector)[0])
        predicted_eis = min(max(predicted_eis, 0.0), 1.0)

        # 4. Resolution time (expm1 log output)
        predicted_res_log = float(self.model_res.predict(feat_vector)[0])
        predicted_res_min = float(np.expm1(predicted_res_log))
        predicted_res_min = max(0.0, predicted_res_min)

        return {
            "requires_road_closure": road_closure_pred,
            "requires_road_closure_prob": round(road_closure_prob, 4),
            "priority_encoded": priority_pred,
            "priority": "High" if priority_pred == 1 else "Low",
            "priority_prob": round(priority_prob, 4),
            "event_impact_score": round(predicted_eis, 4),
            "resolution_time_min": round(predicted_res_min, 2)
        }
