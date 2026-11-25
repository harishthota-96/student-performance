from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
reg = joblib.load(ROOT / "models" / "score_regressor.joblib")
clf = joblib.load(ROOT / "models" / "pass_classifier.joblib")

def predict_student(attendance_pct, study_hours_per_day, assignments_completed,
                    previous_score, internet_quality="Average", course="CSE"):
    sample = pd.DataFrame([{
        "attendance_pct": attendance_pct,
        "study_hours_per_day": study_hours_per_day,
        "assignments_completed": assignments_completed,
        "previous_score": previous_score,
        "internet_quality": internet_quality,
        "course": course
    }])
    score = float(reg.predict(sample)[0])
    passed = int(clf.predict(sample)[0])
    return {"predicted_final_score": round(score, 2),
            "predicted_result": "Pass" if passed else "Fail"}

if __name__ == "__main__":
    print(predict_student(85, 4, 8, 72, "Good", "CSE"))
