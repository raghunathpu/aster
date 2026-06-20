"""
ASTER - Training Entry Point
=============================
Run this once to train the model and generate all artefacts.

Usage:
    python train.py
    python train.py --data data/bengaluru_traffic_events.csv
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.modeling.train import train_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTER model training")
    parser.add_argument(
        "--data",
        default=os.path.join(ROOT, "data", "bengaluru_traffic_events.csv"),
        help="Path to the raw events CSV",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(ROOT, "models"),
        help="Directory to save trained artefacts",
    )
    parser.add_argument(
        "--assets-dir",
        default=os.path.join(ROOT, "assets"),
        help="Directory to save evaluation plots",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] Data file not found: {args.data}")
        sys.exit(1)

    print("=" * 60)
    print("  ASTER - Adaptive Smart Traffic Event Response")
    print("  Model Training Pipeline")
    print("=" * 60)

    train_pipeline(args.data, args.model_dir, args.assets_dir)

    from src.modeling.train_lgb import train_lgb_pipeline
    print("\n" + "=" * 60)
    print("  Training LightGBM Cascading Models")
    print("=" * 60)
    train_lgb_pipeline(args.data, args.model_dir)

    print("\n" + "=" * 60)
    print("  Next step: streamlit run app/aster_app.py")
    print("=" * 60)
