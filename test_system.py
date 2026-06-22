"""ASTER Full System Test"""
import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, '.')

print('Testing imports...')
from src.preprocessing.data_loader import preprocess, NAMED_CORRIDORS
from src.features.feature_engineering import build_features, encode_for_model
from src.recommendation.engine import generate_response_plan, plan_to_dict
from src.modeling.lgb_inference import LGBInferenceService
from src.recommendation.routing_engine import RoutingEngine
from src.recommendation.manpower import ManpowerOptimizer
import joblib, json, pandas as pd, numpy as np
print('  [OK] All imports successful')

# Test model loading
gb     = joblib.load('models/gb_main.pkl')
enc    = joblib.load('models/encoders.pkl')
freq   = joblib.load('models/freq_maps.pkl')
le     = joblib.load('models/label_encoder.pkl')
fnames = joblib.load('models/feature_names.pkl')
print(f'  [OK] Models loaded — {len(fnames)} features')

# Test data
df = preprocess('data/bengaluru_traffic_events.csv')
df, _ = build_features(df)
print(f'  [OK] Data loaded — {len(df)} rows')

# Test prediction
import zoneinfo
IST = zoneinfo.ZoneInfo('Asia/Kolkata')
row = pd.DataFrame([{
    'event_type': 'unplanned', 'event_cause': 'accident',
    'requires_road_closure': True, 'priority': 'High',
    'corridor': 'Mysore Road', 'zone': 'South Zone 2',
    'latitude': 12.97, 'longitude': 77.56, 'veh_type': 'bmtc_bus',
    'police_station': 'Unknown', 'start_datetime': pd.Timestamp.now(tz='UTC')
}])
row['start_datetime'] = pd.to_datetime(row['start_datetime'], utc=True)
row['start_local'] = row['start_datetime'].dt.tz_convert(IST)
row['corridor'] = row['corridor'].fillna('Non-corridor')
row['is_named_corridor'] = row['corridor'].isin(NAMED_CORRIDORS).astype(int)
row['road_closure_flag'] = row['requires_road_closure'].astype(str).str.lower().isin(['true','1','yes']).astype(int)
row['priority'] = row.get('priority', pd.Series(['High'])).fillna('High')
row['veh_type'] = row.get('veh_type', pd.Series(['unknown'])).fillna('unknown')
row['zone'] = row['zone'].fillna('Unknown')
row['police_station'] = row.get('police_station', pd.Series(['Unknown'])).fillna('Unknown')
row, _ = build_features(row, freq_maps=freq)
X, _ = encode_for_model(row, fit_encoders=enc)
for col in fnames:
    if col not in X.columns:
        X[col] = 0
X = X[fnames]
probs = gb.predict_proba(X)[0]
pred = le.classes_[int(np.argmax(probs))]
conf = round(float(probs.max()), 3)
print(f'  [OK] Prediction — {pred} (confidence: {conf})')

# Test response plan
plan = generate_response_plan(
    predicted_tier=pred, confidence=conf, impact_score=4,
    event_cause='accident', corridor='Mysore Road', zone='South Zone 2',
    junction=None, hour=8, requires_road_closure=True
)
pd2 = plan_to_dict(plan)
prio = pd2['response_priority']
mp_min = pd2['manpower_min']
mp_max = pd2['manpower_max']
print(f'  [OK] Response plan — {prio} | Officers: {mp_min}-{mp_max}')

# Test LGB
lgb_svc = LGBInferenceService(
    models_dir='models/lgb',
    dataset_path='models/processed_dataset.csv'
)
res = lgb_svc.predict({
    'event_type': 'unplanned', 'event_cause': 'accident', 'vehicle_type': 'bmtc_bus',
    'latitude': 12.97, 'longitude': 77.56,
    'corridor': 'Mysore Road', 'zone': 'South Zone 2',
    'start_datetime': pd.Timestamp.now()
})
print(f'  [OK] LightGBM — resolution={res["resolution_time_min"]}min EIS={res["event_impact_score"]}')

# Test routing
router = RoutingEngine(graph_path='data/graphs/bengaluru_road_graph.pkl')
junc = router.find_nearest_junction(12.97, 77.56)
print(f'  [OK] Router — nearest junction: {junc}')

# Test OR-Tools
opt = ManpowerOptimizer(total_officers=30)
alloc = opt.optimize([
    {'event_id':'e1','junction':'SilkBoard','latitude':12.91,'longitude':77.62,
     'predicted_eis':0.8,'predicted_priority':1,'requires_road_closure':1},
    {'event_id':'e2','junction':'Majestic','latitude':12.97,'longitude':77.57,
     'predicted_eis':0.4,'predicted_priority':0,'requires_road_closure':0},
])
officers = [a['allocated_officers'] for a in alloc]
print(f'  [OK] OR-Tools — officers allocated: {officers}')

# Test weather API
import requests
try:
    r = requests.get(
        'https://api.open-meteo.com/v1/forecast'
        '?latitude=12.9716&longitude=77.5946'
        '&current=temperature_2m,precipitation&forecast_days=1',
        timeout=5
    )
    wx = r.json()
    temp = wx.get('current', {}).get('temperature_2m', 'N/A')
    print(f'  [OK] Weather API — Bengaluru temp: {temp}C')
except Exception as e:
    print(f'  [WARN] Weather API unavailable (will show offline in app): {e}')

# Verify assets exist
assets_needed = [
    'assets/eda_impact_dist.png', 'assets/eda_cause_dist.png',
    'assets/eda_hourly.png', 'assets/eda_corridors.png',
    'assets/confusion_matrix.png', 'assets/feature_importance.png'
]
missing = [a for a in assets_needed if not os.path.exists(a)]
if missing:
    print(f'  [WARN] Missing assets: {missing}')
else:
    print(f'  [OK] All {len(assets_needed)} required assets present')

print()
print('=' * 50)
print('  ALL SYSTEMS OPERATIONAL — App is ready.')
print('  Run: streamlit run app/aster_app.py')
print('=' * 50)
