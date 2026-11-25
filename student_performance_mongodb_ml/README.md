# Student Performance Prediction using MongoDB, PyMongo and Machine Learning

## Overview
A reproducible academic project that stores student records in MongoDB through PyMongo and trains two Random Forest models:
- **Regression:** predicts a student's final assessment score (0–100).
- **Classification:** predicts whether the student passes (1) or fails (0), using a 40/100 pass threshold.

The bundled CSV is a **synthetic teaching dataset** generated with a fixed random seed. It is not real student or institutional data. The reported metrics are calculated from this exact CSV with an 80/20 train/test split and random state 42.

## Requirements
- Python 3.10+
- MongoDB Community Server running locally, or a MongoDB Atlas connection string
- pip

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```
Edit `.env` and set `MONGODB_URI` if your MongoDB connection is not local.

## Run
From the project root:
```powershell
python src/main.py all
```
This imports/upserts the CSV into `student_performance_db.students`, then trains and saves both models. If MongoDB is unavailable during training, the training script falls back to the bundled CSV; the import step itself requires MongoDB.

Train separately:
```powershell
python src/main.py train
```
Predict using the saved models:
```powershell
python src/predict.py
```

## Database design
Database: `student_performance_db`  
Collection: `students`  
Unique key: `student_id`

Each document contains attendance percentage, study hours, assignments completed, previous score, internet quality, course, final score, and pass label.

## Evaluation
See `reports/evaluation_metrics.json` and the project report PDF. Accuracy, precision, recall and F1-score are classification metrics. R² and MAE are regression metrics; they are not interchangeable.

## Project structure
- `data/` — reproducible synthetic dataset
- `src/` — MongoDB import/export, training and prediction code
- `models/` — trained model artifacts
- `reports/` — evaluation results and project report
- `project_report.md` — editable report source

## Notes
- The model is an academic demonstration, not a validated student assessment or admissions tool.
- For real deployment, use consented real data, audit for bias, validate on a separate cohort, and protect personally identifiable information.


## Run the Streamlit web application

After completing the setup and installing requirements, make sure the trained model files exist in `models/`.

Start the dashboard from the project root:

```powershell
streamlit run app.py
```

Streamlit opens the dashboard in your browser (usually at `http://localhost:8501`). It includes:
- Project overview and dataset summary
- Student performance prediction form
- Student dataset viewer and CSV download
- Model evaluation metrics and confusion matrix

The dashboard attempts to read student records from MongoDB. If MongoDB is unavailable, it displays the bundled CSV dataset instead. Prediction uses the saved Random Forest models.
