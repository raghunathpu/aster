"""
ASTER – Utility Helpers
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def data_path(filename: str = "bengaluru_traffic_events.csv") -> str:
    return os.path.join(ROOT, "data", filename)


def model_dir() -> str:
    return os.path.join(ROOT, "models")


def assets_dir() -> str:
    return os.path.join(ROOT, "assets")


TIER_COLORS = {
    "Low": "#22C55E",
    "Medium": "#F59E0B",
    "High": "#EF4444",
}

TIER_EMOJI = {
    "Low": "🟢",
    "Medium": "🟡",
    "High": "🔴",
}

CAUSE_LABELS = {
    "vehicle_breakdown": "Vehicle Breakdown",
    "accident": "Accident",
    "construction": "Construction",
    "water_logging": "Water Logging",
    "pot_holes": "Potholes",
    "tree_fall": "Tree Fall",
    "public_event": "Public Event",
    "procession": "Procession / Rally",
    "protest": "Protest",
    "vip_movement": "VIP Movement",
    "road_conditions": "Road Conditions",
    "congestion": "General Congestion",
    "others": "Others",
    "fog_visibility": "Fog / Low Visibility",
    "debris": "Debris on Road",
    "unknown": "Unknown",
}

CORRIDORS = [
    "Non-corridor", "Mysore Road", "Bellary Road 1", "Bellary Road 2",
    "Tumkur Road", "Hosur Road", "ORR North 1", "Old Madras Road",
    "Magadi Road", "ORR East 1", "ORR North 2", "Bannerghata Road",
    "ORR East 2", "West of Chord Road", "ORR West 1", "CBD 2",
    "Hennur Main Road", "IRR(Thanisandra road)", "Varthur Road",
    "Old Airport Road",
]

ZONES = [
    "Central Zone 1", "Central Zone 2",
    "North Zone 1", "North Zone 2",
    "South Zone 1", "South Zone 2",
    "East Zone 1", "East Zone 2",
    "West Zone 1", "West Zone 2",
    "Unknown",
]

VEHICLE_TYPES = [
    "unknown", "heavy_vehicle", "bmtc_bus", "ksrtc_bus",
    "private_bus", "lcv", "truck", "private_car", "taxi", "auto", "others",
]
