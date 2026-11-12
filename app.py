"""Streamlit demo for the Aiqon shipment ETA proof-of-concept."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import json
from src.features import build_features

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "shipments_synthetic.csv"
ART = ROOT / "artifacts"

st.set_page_config(
    page_title="Aiqon | Shipment Intelligence",
    page_icon="🚢",
    layout="wide",
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    return (
        joblib.load(ART / "eta_regressor.joblib"),
        joblib.load(ART / "delay_classifier.joblib"),
    )

data = load_data()
regressor, classifier = load_models()

st.title("🚢 Aiqon | Shipment Intelligence")
st.caption(
    "Shipment ETA and delay-propagation proof-of-concept · "
    "Synthetic demonstration data · Prediction point: after customs clearance"
)

with st.sidebar:
    st.header("Shipment selection")
    selected_id = st.selectbox("Shipment ID", data["shipment_id"].tolist())
    st.divider()
    st.caption("Model scope")
    st.write("• Predict final delivery delay")
    st.write("• Estimate probability of delay greater than 24 hours")
    st.write("• Explain influential shipment signals")
    st.divider()
    st.caption("This demo uses a reproducible synthetic dataset. It is not an operational logistics forecast.")

row = data.loc[data["shipment_id"] == selected_id].iloc[[0]]
features = build_features(row)
predicted_delay = float(regressor.predict(features)[0])
delay_probability = float(classifier.predict_proba(features)[0, 1])
scheduled_eta = pd.to_datetime(row.iloc[0]["scheduled_delivery"])
predicted_eta = scheduled_eta + pd.Timedelta(hours=predicted_delay)

is_high_risk = delay_probability >= 0.5
status = "High delay risk" if is_high_risk else "Lower delay risk"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted final delay", f"{predicted_delay:.1f} hours")
c2.metric("Probability of >24h delay", f"{delay_probability:.0%}")
c3.metric("Predicted final ETA", predicted_eta.strftime("%d %b %Y, %H:%M"))
c4.metric("Risk category", status)

st.divider()
left, right = st.columns([1.05, 1])

with left:
    st.subheader("Shipment journey")
    st.write(f"**{row.iloc[0]['origin']} → {row.iloc[0]['destination']}**")
    st.write(f"Route region: {row.iloc[0]['route_region']} · Planned transit: {int(row.iloc[0]['transit_days'])} days")
    stage_data = pd.DataFrame({
        "Stage": ["Origin departure", "Port processing", "Customs clearance"],
        "Observed delay (hours)": [
            row.iloc[0]["origin_departure_delay_hours"],
            row.iloc[0]["port_processing_delay_hours"],
            row.iloc[0]["customs_delay_hours"],
        ],
    })
    st.dataframe(stage_data, hide_index=True, use_container_width=True)
    st.write(f"Port congestion index: **{row.iloc[0]['port_congestion_index']:.2f}**")
    st.write(f"Weather severity index: **{row.iloc[0]['weather_severity_index']:.2f}**")

with right:
    st.subheader("Prediction interpretation")
    st.write(
        f"The model estimates final delivery to be **{predicted_delay:.1f} hours** "
        f"away from the scheduled delivery time."
    )
    st.progress(min(max(delay_probability, 0.0), 1.0))
    st.caption(f"Estimated probability of exceeding the 24-hour delay threshold: {delay_probability:.1%}")
    st.info(
        "The probability is a model estimate, not a guarantee. "
        "The classifier uses a 0.50 decision threshold in this demonstration."
    )

st.divider()
st.subheader("What drives delay predictions?")
importance_path = ART / "feature_importance.csv"
importance = pd.read_csv(importance_path).head(8)
st.bar_chart(importance.set_index("feature")["importance_mean"])
st.caption(
    "Global permutation importance: increase in held-out regression error when a feature is shuffled. "
    "It measures predictive contribution, not causation."
)

with st.expander("View model evaluation"):
    metrics = json.loads((ART / "metrics.json").read_text())
    st.write(f"Dataset: {metrics['dataset_rows']} records · "
             f"Train: {metrics['train_rows']} · Test: {metrics['test_rows']}")
    st.json(metrics)

with st.expander("Methodology and limitations"):
    st.markdown("""
    - Regression target: actual final delivery minus scheduled final delivery, in hours.
    - Classification target: final delay greater than 24 hours.
    - Prediction point: after customs clearance, before final delivery.
    - Regression: Extra Trees. Classification: Random Forest.
    - Missing numeric readings are median-imputed; categorical readings are most-frequent-imputed.
    - The dataset is synthetic. Strong test metrics reflect simulated relationships and should not be interpreted as real-world performance.
    - The local CLI sensitivity explanation is diagnostic, not causal attribution.
    """)
