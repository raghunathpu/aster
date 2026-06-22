"""
ASTER – Data Loader & Preprocessor
Loads and cleans the Bengaluru traffic event dataset.
"""
import pandas as pd
import numpy as np
import zoneinfo

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

CAUSE_MAP = {
    "Debris": "debris",
    "Fog / Low Visibility": "fog_visibility",
    "test_demo": "others",
}

HIGH_IMPACT_CAUSES = {
    "accident", "construction", "public_event",
    "protest", "procession", "vip_movement", "water_logging",
}

NAMED_CORRIDORS = {
    "Mysore Road", "Bellary Road 1", "Bellary Road 2",
    "Tumkur Road", "Hosur Road", "ORR North 1", "Old Madras Road",
    "Magadi Road", "ORR East 1", "ORR North 2", "Bannerghata Road",
    "ORR East 2", "West of Chord Road", "ORR West 1", "CBD 2",
    "Hennur Main Road", "IRR(Thanisandra road)", "Varthur Road",
    "Old Airport Road",
}


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["start_datetime", "closed_datetime", "created_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def localise_times(df: pd.DataFrame) -> pd.DataFrame:
    if "start_datetime" in df.columns:
        df["start_local"] = df["start_datetime"].dt.tz_convert(IST)
    return df


def clean_causes(df: pd.DataFrame) -> pd.DataFrame:
    df["event_cause"] = (
        df["event_cause"]
        .map(lambda x: CAUSE_MAP.get(x, x) if pd.notna(x) else "unknown")
        .astype(str).str.strip()
    )
    return df


def clean_corridor(df: pd.DataFrame) -> pd.DataFrame:
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df["is_named_corridor"] = df["corridor"].isin(NAMED_CORRIDORS).astype(int)
    return df


def clean_zone(df: pd.DataFrame) -> pd.DataFrame:
    df["zone"] = df["zone"].fillna("Unknown")
    return df


def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    if "start_datetime" in df.columns and "closed_datetime" in df.columns:
        delta = (df["closed_datetime"] - df["start_datetime"]).dt.total_seconds()
        df["duration_minutes"] = np.where(delta > 0, delta / 60.0, np.nan)
    return df


def build_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite impact score and tier using Ground Truth Time-to-Resolution.
    If duration is available:
        < 30 min = Low
        30 - 60 min = Medium
        > 60 min = High
    Fallback heuristic applied only if duration is missing.
    """
    def _compute_tier(row):
        duration = row.get("duration_minutes")
        if pd.notna(duration):
            if duration < 30: return "Low"
            elif duration <= 60: return "Medium"
            else: return "High"
        
        # Fallback to operational heuristics
        score = 1
        closure = str(row.get("requires_road_closure", False)).lower() in ["true", "1", "yes"]
        if closure: score += 2
        if row.get("priority") == "High": score += 1
        if row.get("event_cause") in HIGH_IMPACT_CAUSES: score += 1
        if row.get("is_named_corridor", 0) == 1: score += 1
        
        if score <= 2: return "Low"
        elif score == 3: return "Medium"
        return "High"

    df["impact_tier"] = df.apply(_compute_tier, axis=1)
    
    # Map tier back to a 1-6 score for backwards compatibility with UI
    def _tier_score(t):
        if t == "Low": return 2
        elif t == "Medium": return 3
        return 5
    df["impact_score"] = df["impact_tier"].map(_tier_score)
    return df


def preprocess(path: str) -> pd.DataFrame:
    df = load_raw(path)
    df = parse_datetimes(df)
    df = localise_times(df)
    df = clean_causes(df)
    df = clean_corridor(df)
    df = clean_zone(df)
    df = compute_duration(df)
    df = build_impact_score(df)
    df["road_closure_flag"] = df["requires_road_closure"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    ).astype(int)
    df["priority"] = df["priority"].fillna("Low")
    return df
