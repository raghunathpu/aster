"""
ASTER - Prediction CLI
=======================
Quick command-line test of the inference + recommendation engine.

Usage:
    python predict.py
    python predict.py --cause accident --corridor "Mysore Road" --hour 8 --closure
"""

import argparse
import sys
import pandas as pd
import numpy as np
import joblib
import os

# Ensure windows console can print emojis without crashing
sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.modeling.inference import ASTERPredictor
from src.recommendation.engine import generate_response_plan, plan_to_dict

MODELS_DIR = os.path.join(ROOT, "models")


def run_prediction(args):
    if not os.path.exists(os.path.join(MODELS_DIR, "gb_main.pkl")):
        print("[ERROR] Model not found. Run `python train.py` first.")
        sys.exit(1)

    predictor = ASTERPredictor(MODELS_DIR)

    event = {
        "event_type":          args.event_type,
        "event_cause":         args.cause,
        "requires_road_closure": args.closure,
        "priority":            args.priority,
        "corridor":            args.corridor,
        "zone":                args.zone,
        "latitude":            args.lat,
        "longitude":           args.lon,
        "veh_type":            args.veh_type,
        "police_station":      "Unknown",
    }

    print("\n-- Input Event -----------------------------------------")
    for k, v in event.items():
        print(f"  {k:<30} {v}")

    result     = predictor.predict_single(event)
    top_factors = predictor.get_top_factors(event, top_n=5)

    plan = generate_response_plan(
        predicted_tier=result["predicted_tier"],
        confidence=result["confidence"],
        impact_score=result["impact_score_raw"],
        event_cause=event["event_cause"],
        corridor=event["corridor"],
        zone=event["zone"],
        junction=None,
        hour=args.hour,
        requires_road_closure=args.closure,
    )
    plan_d = plan_to_dict(plan)

    print("\n-- ASTER Prediction ------------------------------------")
    print(f"  Predicted Tier   : {result['predicted_tier']}")
    print(f"  Effective Tier   : {plan_d['effective_tier']}")
    print(f"  Confidence       : {result['confidence']*100:.1f}%")
    print(f"  Impact Score     : {result['impact_score_raw']}/6")
    print(f"  Probabilities    : {result['probabilities']}")

    if plan_d["escalation_triggers"]:
        print(f"\n  ^ Escalation Triggers:")
        for t in plan_d["escalation_triggers"]:
            print(f"    * {t}")

    print("\n-- Response Plan ---------------------------------------")
    print(f"  Response Priority : {plan_d['response_priority']}")
    print(f"  Officers Required : {plan_d['manpower_min']}-{plan_d['manpower_max']}")
    print(f"  Deploy Within     : {plan_d['deployment_time']}")
    print(f"  Barricading       : {plan_d['barricading']}")
    print(f"  Diversion         : {plan_d['diversion_urgency']}")
    print(f"\n  Reasoning: {plan_d['risk_reasoning']}")

    print("\n-- Top Contributing Factors ----------------------------")
    for i, f in enumerate(top_factors, 1):
        print(f"  {i}. {f}")

    print("\n-- Action Checklist ------------------------------------")
    for i, item in enumerate(plan_d["action_items"], 1):
        print(f"  {i:>2}. {item}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTER single-event predictor")
    parser.add_argument("--event-type",  default="unplanned", choices=["unplanned","planned"])
    parser.add_argument("--cause",       default="vehicle_breakdown",
                        choices=["vehicle_breakdown","accident","construction","water_logging",
                                 "pot_holes","tree_fall","public_event","procession","protest",
                                 "vip_movement","road_conditions","congestion","others",
                                 "fog_visibility","debris","unknown"])
    parser.add_argument("--closure",     action="store_true", help="Event requires road closure")
    parser.add_argument("--priority",    default="High", choices=["High","Low"])
    parser.add_argument("--corridor",    default="Mysore Road")
    parser.add_argument("--zone",        default="South Zone 2")
    parser.add_argument("--lat",         type=float, default=12.97)
    parser.add_argument("--lon",         type=float, default=77.56)
    parser.add_argument("--hour",        type=int,   default=8)
    parser.add_argument("--veh-type",    default="bmtc_bus",
                        choices=["unknown","heavy_vehicle","bmtc_bus","ksrtc_bus",
                                 "private_bus","lcv","truck","private_car","taxi","auto","others"])
    args = parser.parse_args()
    run_prediction(args)
