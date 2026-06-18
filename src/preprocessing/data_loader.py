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
    Composite impact score 1-6:
        +2  requires_road_closure == True
        +1  priority == High
        +1  event_cause in high-impact set
        +1  named corridor
    Tier: 1-2=Low, 3=Medium, 4-6=High
    """
    score = pd.Series(1, index=df.index)
    closure = df["requires_road_closure"].astype(str).str.lower().isin(["true", "1", "yes"])
    score += closure.astype(int) * 2
    score += (df["priority"] == "High").astype(int)
    score += df["event_cause"].isin(HIGH_IMPACT_CAUSES).astype(int)
    score += df["is_named_corridor"]
    df["impact_score"] = score.clip(1, 6).astype(int)

    def _tier(s):
        if s <= 2: return "Low"
        elif s == 3: return "Medium"
        return "High"

    df["impact_tier"] = df["impact_score"].map(_tier)
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
