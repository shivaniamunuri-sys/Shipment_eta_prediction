"""Train ETA regression and delay classification models; save metrics and artifacts."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "shipments_synthetic.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

def make_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])

def main():
    df = pd.read_csv(DATA)
    X = build_features(df)
    y_reg = df["final_delay_hours"]
    y_cls = df["delayed_over_24h"]
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_cls, test_size=.25, random_state=42, stratify=y_cls
    )

    reg = Pipeline([
        ("prep", make_preprocessor()),
        ("model", ExtraTreesRegressor(n_estimators=350, min_samples_leaf=2,
                                      max_features=.9, random_state=42, n_jobs=-1)),
    ])
    clf = Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
                                         class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    reg.fit(X_train, yr_train)
    clf.fit(X_train, yc_train)
    pred = reg.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]
    label = (prob >= .5).astype(int)
    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "regression": {
            "MAE_hours": float(mean_absolute_error(yr_test, pred)),
            "RMSE_hours": float(mean_squared_error(yr_test, pred) ** .5),
            "R2": float(r2_score(yr_test, pred)),
        },
        "classification": {
            "accuracy": float(accuracy_score(yc_test, label)),
            "precision": float(precision_score(yc_test, label, zero_division=0)),
            "recall": float(recall_score(yc_test, label, zero_division=0)),
            "F1": float(f1_score(yc_test, label, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(yc_test, prob)),
            "positive_class_threshold_hours": 24,
            "probability_threshold": .5,
        },
    }
    # Global model-agnostic importance in the original feature space.
    importance = permutation_importance(reg, X_test, yr_test, n_repeats=12,
                                        random_state=42, scoring="neg_mean_absolute_error",
                                        n_jobs=-1)
    imp = pd.DataFrame({"feature": X_test.columns,
                        "importance_mean": importance.importances_mean,
                        "importance_std": importance.importances_std})
    imp = imp.sort_values("importance_mean", ascending=False)
    imp.to_csv(ART / "feature_importance.csv", index=False)
    joblib.dump(reg, ART / "eta_regressor.joblib")
    joblib.dump(clf, ART / "delay_classifier.joblib")
    metrics["feature_importance_top5"] = imp.head(5).to_dict(orient="records")
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
