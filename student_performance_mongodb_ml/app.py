from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Student Performance ML Dashboard", page_icon="🎓", layout="wide")

@st.cache_resource
def load_models():
    reg = joblib.load(ROOT / "models" / "score_regressor.joblib")
    clf = joblib.load(ROOT / "models" / "pass_classifier.joblib")
    return reg, clf

@st.cache_data
def load_metrics():
    path = ROOT / "reports" / "evaluation_metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

@st.cache_data
def load_csv():
    return pd.read_csv(ROOT / "data" / "student_performance.csv")

def load_database_records():
    try:
        from mongo_store import export_records
        docs = export_records()
        if docs:
            return pd.DataFrame(docs), "MongoDB"
    except Exception:
        pass
    return load_csv(), "Bundled CSV fallback"

st.title("🎓 Student Performance Prediction")
st.caption("MongoDB + PyMongo + Random Forest Machine Learning")
st.info("Academic demonstration using a synthetic dataset. Predictions are illustrative and should not be used for real academic decisions.")

reg_model, clf_model = load_models()
metrics = load_metrics()
data, data_source = load_database_records()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page", ["Overview", "Predict Student Result", "Student Dataset", "Model Evaluation"])

if page == "Overview":
    st.subheader("Project Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Student Records", f"{len(data):,}")
    c2.metric("Data Source", data_source)
    c3.metric("ML Algorithm", "Random Forest")
    st.markdown("### Dataset preview")
    st.dataframe(data.head(10), use_container_width=True)
    if "passed" in data.columns:
        pass_count = int((data["passed"] == 1).sum())
        fail_count = int((data["passed"] == 0).sum())
        st.markdown("### Pass / Fail distribution")
        st.bar_chart(pd.DataFrame({"Students": [pass_count, fail_count]}, index=["Pass", "Fail"]))
    st.markdown("### Project workflow")
    st.write("1. Student records are stored in MongoDB.")
    st.write("2. PyMongo retrieves the records.")
    st.write("3. Scikit-learn preprocesses numeric and categorical features.")
    st.write("4. Random Forest models predict final score and pass/fail result.")

elif page == "Predict Student Result":
    st.subheader("Enter Student Details")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            attendance = st.slider("Attendance (%)", 0, 100, 75)
            study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
            assignments = st.slider("Assignments completed", 0, 10, 6)
        with col2:
            previous_score = st.slider("Previous score", 0, 100, 60)
            internet = st.selectbox("Internet quality", ["Good", "Average", "Poor"])
            course = st.selectbox("Course", ["CSE", "ECE", "ME", "BBA"])
        submitted = st.form_submit_button("Predict Performance", type="primary")

    if submitted:
        sample = pd.DataFrame([{
            "attendance_pct": attendance,
            "study_hours_per_day": study_hours,
            "assignments_completed": assignments,
            "previous_score": previous_score,
            "internet_quality": internet,
            "course": course
        }])
        predicted_score = float(reg_model.predict(sample)[0])
        predicted_class = int(clf_model.predict(sample)[0])
        st.markdown("### Prediction")
        m1, m2 = st.columns(2)
        m1.metric("Predicted Final Score", f"{predicted_score:.2f} / 100")
        if predicted_class == 1:
            m2.success("Predicted Result: PASS")
        else:
            m2.error("Predicted Result: FAIL")
        st.caption("This is a model estimate, not a guaranteed academic result.")

elif page == "Student Dataset":
    st.subheader("Student Records")
    search = st.text_input("Search by Student ID")
    filtered = data.copy()
    if search and "student_id" in filtered.columns:
        filtered = filtered[filtered["student_id"].astype(str).str.contains(search, case=False, na=False)]
    st.write(f"Showing {len(filtered):,} record(s). Source: **{data_source}**")
    st.dataframe(filtered, use_container_width=True, height=520)
    st.download_button("Download displayed data as CSV", filtered.to_csv(index=False).encode("utf-8"), "student_records.csv", "text/csv")

elif page == "Model Evaluation":
    st.subheader("Model Evaluation Results")
    reg_metrics = metrics.get("regression", {})
    cls_metrics = metrics.get("classification", {})
    a, b = st.columns(2)
    with a:
        st.markdown("#### Regression")
        st.metric("R² Score", f"{reg_metrics.get('r2_score', 0):.4f}")
        st.metric("Mean Absolute Error (MAE)", f"{reg_metrics.get('mae', 0):.4f}")
    with b:
        st.markdown("#### Classification")
        st.metric("Accuracy", f"{cls_metrics.get('accuracy', 0):.2%}")
        st.metric("Precision", f"{cls_metrics.get('precision', 0):.2%}")
        st.metric("Recall", f"{cls_metrics.get('recall', 0):.2%}")
        st.metric("F1 Score", f"{cls_metrics.get('f1_score', 0):.4f}")
    cm = cls_metrics.get("confusion_matrix", metrics.get("confusion_matrix"))
    if cm:
        st.markdown("#### Confusion Matrix")
        st.write("Rows = actual [Fail, Pass]; columns = predicted [Fail, Pass].")
        st.dataframe(pd.DataFrame(cm, index=["Actual Fail", "Actual Pass"], columns=["Predicted Fail", "Predicted Pass"]), use_container_width=True)
    st.caption("Metrics are calculated from the synthetic dataset test split and are for academic demonstration.")

st.sidebar.divider()
st.sidebar.caption("MongoDB database: student_performance_db")
st.sidebar.caption("Collection: students")
