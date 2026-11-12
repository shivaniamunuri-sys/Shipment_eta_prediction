# Aiqon — Shipment ETA & Delay Propagation

A balanced, reproducible proof-of-concept for the Aiqon AI internship take-home assignment.

## What it does

- Predicts final shipment delay in hours.
- Estimates the probability that final delivery will be more than 24 hours late.
- Explains global feature importance and provides a simple shipment-level sensitivity diagnostic.
- Includes a Streamlit dashboard for an interactive demonstration.

## Project structure

```text
aiqon_shipment_eta/
├── app.py
├── requirements.txt
├── README.md
├── run_demo.bat
├── run_demo.sh
├── data/
│   ├── shipments_synthetic.csv
│   └── DATA_DICTIONARY.md
├── src/
│   ├── features.py
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_project.py
└── artifacts/
    ├── eta_regressor.joblib
    ├── delay_classifier.joblib
    ├── metrics.json
    └── feature_importance.csv
```

## Requirements

- Python 3.10 or newer recommended.
- Windows, macOS or Linux.
- Internet access is needed only to install Python dependencies.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the project

Train the models:

```bash
python src/train.py
```

Predict one sample shipment:

```bash
python src/predict.py --shipment-id AQ-0001
```

Launch the dashboard:

```bash
streamlit run app.py
```

Run integrity tests:

```bash
python -m unittest discover -s tests -v
```

On Windows, `run_demo.bat` installs requirements, trains the models and launches Streamlit.

## Modeling choices

- Regression: Extra Trees Regressor, predicting `final_delay_hours`.
- Classification: Random Forest Classifier, predicting `delayed_over_24h`.
- Numeric preprocessing: median imputation.
- Categorical preprocessing: most-frequent imputation and one-hot encoding.
- Evaluation: MAE, RMSE, R², accuracy, precision, recall, F1 and ROC-AUC.
- Explainability: held-out permutation importance and a local one-feature-at-a-time sensitivity check.

## Delay propagation assumptions

The synthetic generator models final delay as a combination of weighted origin, port and customs delays, congestion and weather effects, a congestion-by-upstream-delay interaction, partial recovery and random noise. This makes the propagation mechanism explicit and reproducible. It is a simulation assumption, not a learned causal model.

## Definitions

- Final delay (hours) = actual final delivery timestamp − scheduled final delivery timestamp.
- Delayed shipment = final delay greater than 24 hours.
- Prediction point = after customs clearance and before final delivery.

See `data/DATA_DICTIONARY.md` for field definitions.

## Important limitations

1. The dataset is synthetic (300 rows) and cannot establish real-world performance.
2. Random train/test splitting is used for this compact demonstration; operational validation should be chronological.
3. Feature importance and local sensitivity are predictive diagnostics, not causal explanations.
4. Before deployment, use historical shipment, AIS, port, customs, carrier, route and weather data; calibrate risk probabilities; quantify prediction uncertainty; and monitor drift and missing data.

## Reproducibility

The data generator uses a fixed seed (`42`). Regenerate the CSV with:

```bash
python src/generate_data.py
```

Then retrain using `python src/train.py`. Model artifacts and evaluation metrics are saved in `artifacts/`.
