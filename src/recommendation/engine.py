"""
ASTER - Operational Recommendation Engine
==========================================
Converts a predicted impact tier + event context into a structured,
actionable response plan for Bengaluru Traffic Police operations.

All thresholds are grounded in BTP Standard Operating Procedures (assumed)
and the distribution patterns observed in the training dataset.
"""

from dataclasses import dataclass, field
from typing import Optional, List


# ─────────────────────────────────────────────────────────────
# Response Policy Tables
# ─────────────────────────────────────────────────────────────

# Manpower recommendation: (min_officers, max_officers)
MANPOWER_TABLE = {
    "Low": (1, 2),
    "Medium": (2, 4),
    "High": (4, 8),
}

# Additional officers for specific causes
CAUSE_EXTRA_MANPOWER = {
    "accident": 2,
    "public_event": 3,
    "protest": 3,
    "procession": 2,
    "vip_movement": 4,
    "construction": 1,
    "water_logging": 1,
}

# Barricading intensity
BARRICADE_TABLE = {
    "Low": "None",
    "Medium": "Light (2-4 cones / 1 barrier)",
    "High": "Heavy (full lane closure, 6+ barriers)",
}

# Diversion urgency
DIVERSION_TABLE = {
    "Low": "None required",
    "Medium": "Advisory (notify via PA/WhatsApp broadcast)",
    "High": "Mandatory (activate pre-defined diversion routes)",
}

# Response priority label
RESPONSE_PRIORITY_TABLE = {
    "Low": "P3 - Routine",
    "Medium": "P2 - Elevated",
    "High": "P1 - Critical",
}

# Estimated deployment time
DEPLOYMENT_TIME_TABLE = {
    "Low": "15-30 minutes",
    "Medium": "5-15 minutes",
    "High": "Immediate (< 5 minutes)",
}

# Cause-specific escalation rules
ESCALATION_CAUSES = {"accident", "protest", "vip_movement", "public_event"}

# Peak hour causes that always get an upgrade
PEAK_HOUR_CAUSES = {"vehicle_breakdown", "water_logging", "construction"}

# Known high-stress junctions that trigger +1 tier
HIGH_STRESS_JUNCTIONS = {
    "MekhriCircle", "SilkBoardJunc", "HebbalFlyoverJunc",
    "YeshwanthpuraCircle", "JalahalliCross(SM Circle)",
    "KoramangalaWaterTankJunc", "NagavaraORRJunction",
    "toll gate mysore road",
}

# Named high-traffic corridors
NAMED_CORRIDORS = {
    "Mysore Road", "Bellary Road 1", "Bellary Road 2",
    "Tumkur Road", "Hosur Road", "ORR North 1", "Old Madras Road",
    "Magadi Road", "ORR East 1", "ORR North 2", "Bannerghata Road",
    "ORR East 2", "West of Chord Road", "ORR West 1", "CBD 2",
    "Hennur Main Road", "IRR(Thanisandra road)", "Varthur Road",
    "Old Airport Road",
}


@dataclass
class ResponsePlan:
    """Complete operational response recommendation."""
    # Core prediction
    predicted_tier: str
    effective_tier: str          # after context-based escalation
    confidence: float
    impact_score: int

    # Operational outputs
    manpower_min: int
    manpower_max: int
    barricading: str
    diversion_urgency: str
    response_priority: str
    deployment_time: str

    # Reasoning
    escalation_triggers: List[str]
    action_items: List[str]
    risk_reasoning: str

    # Metadata
    event_cause: str
    corridor: str
    zone: str
    hour: int
    is_peak_hour: bool


def _is_peak_hour(hour: int) -> bool:
    return (7 <= hour <= 10) or (17 <= hour <= 21)


def _tier_to_int(tier: str) -> int:
    return {"Low": 0, "Medium": 1, "High": 2}.get(tier, 1)


def _int_to_tier(i: int) -> str:
    return ["Low", "Medium", "High"][max(0, min(2, i))]


def _compute_escalation(
    base_tier: str,
    event_cause: str,
    corridor: str,
    junction: Optional[str],
    hour: int,
    requires_road_closure: bool,
    **kwargs
) -> tuple:
    """
    Apply context-aware escalation logic.
    Returns (effective_tier, [list of escalation trigger descriptions]).
    """
    tier_int = _tier_to_int(base_tier)
    triggers = []

    # Weather Rule: Heavy rain or high multiplier upgrades tier
    if kwargs.get("wx_multiplier", 1.0) >= 1.2:
        tier_int = min(2, tier_int + 1)
        triggers.append(f"Weather Risk: Applied {kwargs.get('wx_multiplier')}x multiplier due to active rain/conditions.")

    # Rule 1: High-stress cause on named corridor -> upgrade
    if event_cause in ESCALATION_CAUSES and corridor in NAMED_CORRIDORS:
        tier_int = min(2, tier_int + 1)
        triggers.append(f"High-impact cause ({event_cause}) on major corridor ({corridor})")

    # Rule 2: Peak hour + road closure -> upgrade
    peak = _is_peak_hour(hour)
    if peak and requires_road_closure:
        tier_int = min(2, tier_int + 1)
        triggers.append("Road closure during peak hours")

    # Rule 3: Junction hotspot
    if junction and junction in HIGH_STRESS_JUNCTIONS:
        tier_int = min(2, tier_int + 1)
        triggers.append(f"Known high-stress junction: {junction}")

    # Rule 4: Peak hour + common disruptive cause
    if peak and event_cause in PEAK_HOUR_CAUSES:
        tier_int = min(2, tier_int + 1)
        triggers.append(f"{event_cause.replace('_',' ').title()} during peak hours")

    effective_tier = _int_to_tier(tier_int)
    return effective_tier, triggers


def _manpower(tier: str, cause: str, road_closure: bool) -> tuple:
    base_min, base_max = MANPOWER_TABLE[tier]
    extra = CAUSE_EXTRA_MANPOWER.get(cause, 0)
    if road_closure:
        extra += 1
    return base_min + extra, base_max + extra


def _action_items(
    tier: str,
    cause: str,
    corridor: str,
    road_closure: bool,
    is_peak: bool,
) -> list:
    items = []

    # Universal
    items.append(f"Dispatch {MANPOWER_TABLE[tier][0]}-{MANPOWER_TABLE[tier][1]} officers to incident site")
    items.append("Log event in ASTER dashboard with timestamp")

    # Tier-specific
    if tier == "High":
        items.append("Activate pre-defined diversion routes for affected corridor")
        items.append("Notify Traffic Control Room (TCR) for coordination")
        items.append("Alert downstream signals for progressive signal timing adjustment")
    if tier == "Medium":
        items.append("Issue advisory via BTP WhatsApp broadcast channel")
        items.append("Monitor situation every 15 minutes for escalation")
    if tier == "Low":
        items.append("Monitor via nearest patrol unit")

    # Cause-specific
    if cause == "accident":
        items.append("Coordinate with BBMP ambulance / CATS for medical response")
        items.append("Preserve accident scene until traffic released by IO")
    if cause in ["public_event", "procession"]:
        items.append("Coordinate with event organisers for crowd and vehicle management")
        items.append("Pre-position barriers 30 min before expected dispersal")
    if cause == "vip_movement":
        items.append("Activate VIP corridor clearance protocol")
        items.append("Station officers at key intersections on the route")
    if cause == "water_logging":
        items.append("Alert BBMP drainage cell for pump deployment")
        items.append("Identify alternate road with no underpasses")
    if cause == "construction":
        items.append("Verify contractor compliance with lane-closure permit")
        items.append("Ensure Automatic Flagger Assistance Device (AFAD) is deployed")
    if cause == "tree_fall":
        items.append("Contact BBMP Tree Officer for immediate clearance")
    if cause == "vehicle_breakdown":
        items.append("Arrange for tow truck if vehicle blocks > 1 lane")

    if road_closure:
        items.append("Set up traffic marshals at closest upstream junction")
        items.append("Activate alternate route signage")

    if is_peak and tier in ["Medium", "High"]:
        items.append("Consider adaptive signal control for upstream signals")

    return items


def _risk_reasoning(tier: str, cause: str, corridor: str, confidence: float, hour: int) -> str:
    time_label = "peak hours" if _is_peak_hour(hour) else "off-peak hours"
    corridor_label = corridor if corridor != "Non-corridor" else "a non-corridor road"

    tier_explanation = {
        "Low": "low operational disruption expected",
        "Medium": "moderate disruption with manageable impact on traffic flow",
        "High": "significant congestion and operational disruption expected",
    }[tier]

    return (
        f"This {cause.replace('_', ' ')} event on {corridor_label} during {time_label} "
        f"is predicted as {tier} impact ({tier_explanation}). "
        f"Model confidence: {confidence * 100:.0f}%."
    )


def generate_response_plan(
    predicted_tier: str,
    confidence: float,
    impact_score: int,
    event_cause: str,
    corridor: str,
    zone: str,
    junction: Optional[str],
    hour: int,
    requires_road_closure: bool,
    top_factors: Optional[list] = None,
    station_capacity: int = 5,
    wx_multiplier: float = 1.0,
) -> ResponsePlan:
    """
    Core function: produce a full ResponsePlan from model output + context.
    """
    requires_road_closure = str(requires_road_closure).lower() in ["true", "1", "yes"]
    is_peak = _is_peak_hour(hour)

    effective_tier, triggers = _compute_escalation(
        predicted_tier, event_cause, corridor, junction, hour, requires_road_closure, wx_multiplier=wx_multiplier
    )

    mp_min, mp_max = _manpower(effective_tier, event_cause, requires_road_closure)
    action_items = _action_items(effective_tier, event_cause, corridor, requires_road_closure, is_peak)
    reasoning = _risk_reasoning(effective_tier, event_cause, corridor, confidence, hour)
    
    # Resource Optimization Constraint
    if mp_max > station_capacity:
        triggers.append(f"Capacity Overflow: {mp_max} officers required but station limit is {station_capacity}.")
        action_items.insert(0, f"🚨 WARNING: Required manpower ({mp_max}) exceeds station capacity ({station_capacity}). Request backup from adjacent station.")

    return ResponsePlan(
        predicted_tier=predicted_tier,
        effective_tier=effective_tier,
        confidence=confidence,
        impact_score=impact_score,
        manpower_min=mp_min,
        manpower_max=mp_max,
        barricading=BARRICADE_TABLE[effective_tier],
        diversion_urgency=DIVERSION_TABLE[effective_tier],
        response_priority=RESPONSE_PRIORITY_TABLE[effective_tier],
        deployment_time=DEPLOYMENT_TIME_TABLE[effective_tier],
        escalation_triggers=triggers,
        action_items=action_items,
        risk_reasoning=reasoning,
        event_cause=event_cause,
        corridor=corridor,
        zone=zone,
        hour=hour,
        is_peak_hour=is_peak,
    )


def plan_to_dict(plan: ResponsePlan) -> dict:
    """Convert ResponsePlan to a plain dict for serialization / display."""
    return {
        "predicted_tier": plan.predicted_tier,
        "effective_tier": plan.effective_tier,
        "confidence": plan.confidence,
        "impact_score": plan.impact_score,
        "manpower_min": plan.manpower_min,
        "manpower_max": plan.manpower_max,
        "barricading": plan.barricading,
        "diversion_urgency": plan.diversion_urgency,
        "response_priority": plan.response_priority,
        "deployment_time": plan.deployment_time,
        "escalation_triggers": plan.escalation_triggers,
        "action_items": plan.action_items,
        "risk_reasoning": plan.risk_reasoning,
        "event_cause": plan.event_cause,
        "corridor": plan.corridor,
        "zone": plan.zone,
        "hour": plan.hour,
        "is_peak_hour": plan.is_peak_hour,
    }


# ──────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plan = generate_response_plan(
        predicted_tier="Medium",
        confidence=0.72,
        impact_score=3,
        event_cause="water_logging",
        corridor="Mysore Road",
        zone="South Zone 2",
        junction="SilkBoardJunc",
        hour=8,
        requires_road_closure=False,
    )
    import json
    print(json.dumps(plan_to_dict(plan), indent=2))
