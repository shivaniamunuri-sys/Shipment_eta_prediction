"""Predict final ETA and delay risk for one row in the shipment CSV."""
from pathlib import Path
import argparse
import joblib
import pandas as pd
from features import build_features

ROOT = Path(__file__).resolve().parents[1]

def explain_one(model, row, feature_columns):
    """Simple one-feature-at-a-time sensitivity, not causal attribution."""
    baseline = float(model.predict(row)[0])
    impacts = []
    medians = pd.read_csv(ROOT / "data" / "shipments_synthetic.csv")
    for col in feature_columns:
        altered = row.copy()
        if pd.api.types.is_numeric_dtype(medians[col]):
            altered[col] = medians[col].median()
        else:
            altered[col] = medians[col].mode(dropna=True).iloc[0]
        changed = float(model.predict(altered)[0])
        impacts.append({"feature": col, "baseline_prediction_hours": baseline,
                        "prediction_after_baseline_replacement_hours": changed,
                        "sensitivity_hours": baseline - changed})
    return sorted(impacts, key=lambda x: abs(x["sensitivity_hours"]), reverse=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shipment-id", default="AQ-0001")
    args = parser.parse_args()
    data = pd.read_csv(ROOT / "data" / "shipments_synthetic.csv")
    match = data[data.shipment_id == args.shipment_id]
    if match.empty:
        raise SystemExit(f"Shipment {args.shipment_id} not found")
    row = match.iloc[[0]]
    X = build_features(row)
    reg = joblib.load(ROOT / "artifacts" / "eta_regressor.joblib")
    clf = joblib.load(ROOT / "artifacts" / "delay_classifier.joblib")
    delay = float(reg.predict(X)[0])
    risk = float(clf.predict_proba(X)[0, 1])
    scheduled = pd.to_datetime(row.iloc[0]["scheduled_delivery"])
    predicted_eta = scheduled + pd.Timedelta(hours=delay)
    print(f"Shipment: {args.shipment_id}")
    print(f"Predicted final delay: {delay:.1f} hours")
    print(f"Probability of >24h delay: {risk:.1%}")
    print(f"Predicted final ETA: {predicted_eta.isoformat()}")
    print("\nLargest one-feature sensitivities (hours):")
    for item in explain_one(reg, X, X.columns)[:5]:
        print(f"- {item['feature']}: {item['sensitivity_hours']:+.1f}")

if __name__ == "__main__":
    main()
