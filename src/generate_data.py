"""Regenerate the reproducible synthetic dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(output_path="data/shipments_synthetic.csv", n=300, seed=42):
    rng = np.random.default_rng(seed)
    routes = [
        ("Shanghai", "Rotterdam", 27, "Asia-Europe"),
        ("Singapore", "Hamburg", 24, "Asia-Europe"),
        ("Mumbai", "Jebel Ali", 6, "Asia-Middle East"),
        ("Los Angeles", "Long Beach", 3, "North America"),
        ("Santos", "Rotterdam", 22, "South America-Europe"),
        ("Busan", "Antwerp", 25, "Asia-Europe"),
    ]
    rows, base = [], pd.Timestamp("2025-01-01 08:00:00")
    for i in range(n):
        origin, destination, transit_days, region = routes[int(rng.integers(0, len(routes)))]
        sd = base + pd.Timedelta(days=int(rng.integers(0, 240)), hours=int(rng.integers(0, 24)))
        sp = sd + pd.Timedelta(days=transit_days)
        sc = sp + pd.Timedelta(hours=float(rng.uniform(8, 48)))
        sf = sc + pd.Timedelta(days=float(rng.uniform(1, 4)))
        congestion = float(rng.beta(2.0, 2.2))
        weather = float(rng.beta(1.7, 3.0))
        peak = int(sd.month in [6, 7, 8, 11, 12])
        origin_d = float(max(-4, rng.normal(3 + 8*weather + 4*peak, 9)))
        port_d = float(max(0, rng.gamma(2, 5) + 28*congestion + 7*weather + 5*peak))
        customs_d = float(max(0, rng.gamma(1.6, 3.5) + 15*congestion + 3*weather))
        interaction = congestion * (origin_d + port_d)
        recovery = float(rng.uniform(.05, .30) * max(origin_d, 0))
        final_d = float(max(-8, .82*origin_d + .88*port_d + .96*customs_d
                            + 10*congestion + 6*weather + .12*interaction
                            - recovery + rng.normal(0, 5)))
        od = origin_d if rng.random() > .045 else np.nan
        pdly = port_d if rng.random() > .075 else np.nan
        cd = customs_d if rng.random() > .06 else np.nan
        rows.append({
            "shipment_id": f"AQ-{i+1:04d}", "origin": origin, "destination": destination,
            "route_region": region, "scheduled_departure": sd.isoformat(),
            "scheduled_port_arrival": sp.isoformat(), "scheduled_customs_clearance": sc.isoformat(),
            "scheduled_delivery": sf.isoformat(),
            "actual_departure": (sd + pd.Timedelta(hours=origin_d)).isoformat(),
            "actual_port_arrival": (sp + pd.Timedelta(hours=origin_d*.82 + port_d*.25)).isoformat(),
            "actual_customs_clearance": (sc + pd.Timedelta(hours=origin_d*.82 + port_d*.88 + customs_d)).isoformat(),
            "actual_delivery": (sf + pd.Timedelta(hours=final_d)).isoformat(),
            "transit_days": transit_days, "origin_departure_delay_hours": od,
            "port_processing_delay_hours": pdly, "customs_delay_hours": cd,
            "port_congestion_index": congestion, "weather_severity_index": weather,
            "peak_season": peak, "cumulative_observed_delay_hours": np.nansum([od, pdly, cd]),
            "port_delay_congestion_interaction": (pdly if not np.isnan(pdly) else 0)*congestion,
            "weather_congestion_interaction": weather*congestion,
            "final_delay_hours": final_d, "delayed_over_24h": int(final_d > 24),
        })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

if __name__ == "__main__":
    generate()
