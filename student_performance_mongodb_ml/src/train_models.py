from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
from mongo_store import export_records

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student_performance.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

def load_data():
    try:
        records = export_records()
        if records:
            return pd.DataFrame(records)
    except Exception as exc:
        print(f"MongoDB unavailable ({exc}); using bundled CSV for training.")
    return pd.read_csv(DATA)

def main():
    df = load_data()
    features = ["attendance_pct", "study_hours_per_day", "assignments_completed",
                "previous_score", "internet_quality", "course"]
    numeric = ["attendance_pct", "study_hours_per_day", "assignments_completed", "previous_score"]
    categorical = ["internet_quality", "course"]
    X, y_reg, y_cls = df[features], df["final_score"], df["passed"]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    Xtr, Xte, yrtr, yrte = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg = Pipeline([("preprocess", preprocessor),
                    ("model", RandomForestRegressor(n_estimators=250, max_depth=12,
                      min_samples_leaf=2, random_state=42, n_jobs=-1))])
    reg.fit(Xtr, yrtr)
    rp = reg.predict(Xte)
    reg_results = {"r2_score": float(r2_score(yrte, rp)),
                   "mae": float(mean_absolute_error(yrte, rp))}
    Xtrc, Xtec, ytrc, ytec = train_test_split(X, y_cls, test_size=0.2,
                                              random_state=42, stratify=y_cls)
    clf = Pipeline([("preprocess", preprocessor),
                    ("model", RandomForestClassifier(n_estimators=250, max_depth=12,
                      min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))])
    clf.fit(Xtrc, ytrc)
    cp = clf.predict(Xtec)
    cls_results = {"accuracy": float(accuracy_score(ytec, cp)),
                   "precision": float(precision_score(ytec, cp, zero_division=0)),
                   "recall": float(recall_score(ytec, cp, zero_division=0)),
                   "f1_score": float(f1_score(ytec, cp, zero_division=0)),
                   "confusion_matrix": confusion_matrix(ytec, cp).tolist()}
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(reg, MODELS / "score_regressor.joblib")
    joblib.dump(clf, MODELS / "pass_classifier.joblib")
    results = {"regression": reg_results, "classification": cls_results,
               "dataset_rows": int(len(df)), "test_rows": int(len(Xte))}
    (REPORTS / "evaluation_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
