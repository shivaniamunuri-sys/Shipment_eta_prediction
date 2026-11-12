"""Feature definitions shared by training and inference."""
NUMERIC_FEATURES = [
    "transit_days",
    "origin_departure_delay_hours",
    "port_processing_delay_hours",
    "customs_delay_hours",
    "port_congestion_index",
    "weather_severity_index",
    "peak_season",
    "cumulative_observed_delay_hours",
    "port_delay_congestion_interaction",
    "weather_congestion_interaction",
]
CATEGORICAL_FEATURES = ["origin", "destination", "route_region"]
TARGET_REGRESSION = "final_delay_hours"
TARGET_CLASSIFICATION = "delayed_over_24h"

def build_features(frame):
    """Return only fields available at the customs-clearance prediction point."""
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return frame[cols].copy()
