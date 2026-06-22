import streamlit as st
import pandas as pd
import math

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_location(query):
    """Geocode a search query to a list of matching addresses and coordinates in Bengaluru."""
    if not query:
        return []
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="aster_traffic_triage_app")
        full_query = query + ", Bengaluru" if "bengaluru" not in query.lower() else query
        
        # Try strictly bounded search first
        results = geolocator.geocode(
            full_query,
            exactly_one=False,
            limit=5,
            viewbox=((12.7, 77.3), (13.3, 77.9)),
            bounded=True
        )
        
        # Fallback to biased (non-bounded) search if no results found
        if not results:
            results = geolocator.geocode(
                full_query,
                exactly_one=False,
                limit=5,
                viewbox=((12.7, 77.3), (13.3, 77.9)),
                bounded=False
            )
            
        if results:
            return [{"address": r.address, "lat": r.latitude, "lon": r.longitude} for r in results]
    except Exception:
        pass
    return []

def match_corridor_by_name(address_text):
    """Attempts to match an address name to a known corridor keyword."""
    if not address_text:
        return None
    address_lower = address_text.lower()
    named_corridors_keywords = {
        "mysore road": "Mysore Road",
        "tumkur road": "Tumkur Road",
        "hosur road": "Hosur Road",
        "old madras road": "Old Madras Road",
        "magadi road": "Magadi Road",
        "bannerghata road": "Bannerghata Road",
        "hennur main road": "Hennur Main Road",
        "hennur road": "Hennur Main Road",
        "varthur road": "Varthur Road",
        "old airport road": "Old Airport Road",
        "west of chord road": "West of Chord Road",
        "chord road": "West of Chord Road",
        "thanisandra road": "IRR(Thanisandra road)",
    }
    for kw, corr in named_corridors_keywords.items():
        if kw in address_lower:
            return corr
            
    # Check for segment-grouped corridors
    if "bellary road" in address_lower:
        return "Bellary Road"
    if "orr east" in address_lower:
        return "ORR East"
    if "orr north" in address_lower:
        return "ORR North"
        
    return None

def detect_corridor_and_zone_py(lat, lon, df, address_text=None):
    """Automatically detect the closest corridor and zone for the given coordinates."""
    temp_df = df.dropna(subset=["latitude", "longitude"])
    if temp_df.empty:
        return "Unknown", "Non-corridor"
    
    # 1. Zone detection (nearest neighbor based on all valid non-unknown zones)
    zone_df = temp_df[temp_df["zone"].notna() & (temp_df["zone"] != "Unknown")]
    if not zone_df.empty:
        z_dists = (zone_df["latitude"] - lat)**2 + (zone_df["longitude"] - lon)**2
        detected_zone = zone_df.loc[z_dists.idxmin(), "zone"]
    else:
        detected_zone = "Unknown"
        
    # 2. Corridor detection
    detected_corridor = None
    if address_text:
        # Check if the address explicitly mentions a corridor
        detected_corridor = match_corridor_by_name(address_text)
        
    corr_df = temp_df[temp_df["corridor"].notna() & (temp_df["corridor"] != "Non-corridor")]
    
    if detected_corridor:
        # Resolve segment-grouped corridors (Bellary Road, ORR East, ORR North) to nearest segment
        if detected_corridor in ["Bellary Road", "ORR East", "ORR North"]:
            seg_df = corr_df[corr_df["corridor"].str.contains(detected_corridor, case=False)]
            if not seg_df.empty:
                s_dists = (seg_df["latitude"] - lat)**2 + (seg_df["longitude"] - lon)**2
                detected_corridor = seg_df.loc[s_dists.idxmin(), "corridor"]
    else:
        # Distance-based detection for named corridors
        if not corr_df.empty:
            c_dists = (corr_df["latitude"] - lat)**2 + (corr_df["longitude"] - lon)**2
            min_c_dist = c_dists.min()
            # 0.008 degrees squared is ~800 meters threshold
            if min_c_dist < 0.000064:
                detected_corridor = corr_df.loc[c_dists.idxmin(), "corridor"]
            else:
                detected_corridor = "Non-corridor"
        else:
            detected_corridor = "Non-corridor"
            
    return detected_zone, detected_corridor
