# Student Performance Prediction Using MongoDB, PyMongo and Machine Learning

## 1. Introduction
Educational institutions collect information about attendance, study habits, assignment completion and previous academic performance. Analysing these factors can help identify patterns associated with final scores and pass/fail outcomes. This project demonstrates a small end-to-end data science workflow: student records are stored in MongoDB, accessed through PyMongo, preprocessed with scikit-learn, and used to train machine learning models.

The bundled dataset is synthetic and generated for academic demonstration. It does not represent real students or any institution.

## 2. Project Objectives
1. Create a MongoDB database and a `students` collection.
2. Connect Python to MongoDB using PyMongo.
3. import and retrieve student records.
4. Clean and preprocess numerical and categorical features.
5. Train a regression model to estimate final scores.
6. Train a classification model to predict pass/fail.
7. Evaluate the models using the requested metrics.

## 3. Dataset Description
The dataset contains **1200 records** and nine columns.

| Field | Description | Type |
|---|---|---|
| student_id | Unique synthetic student key | String |
| attendance_pct | Attendance percentage | Numeric |
| study_hours_per_day | Average daily study hours | Numeric |
| assignments_completed | Completed assignments (0–10) | Integer |
| previous_score | Previous assessment score | Numeric |
| internet_quality | Good, Average or Poor | Categorical |
| course | CSE, ECE, ME or BBA | Categorical |
| final_score | Final score from 0 to 100 | Numeric target |
| passed | 1 if final_score is at least 40, otherwise 0 | Binary target |

The dataset was created using NumPy with random seed 42 so that results can be reproduced. It is a synthetic dataset with deliberately generated relationships; results should not be interpreted as evidence about real students.

## 4. MongoDB and PyMongo Implementation
The database is named `student_performance_db` and the collection is named `students`. PyMongo's `MongoClient` establishes the connection. A unique index is created on `student_id`. Each CSV row is converted to a Python dictionary and inserted using `update_one(..., upsert=True)`. Upsert makes repeated imports safe: existing student records are updated instead of duplicated.

The project reads records from MongoDB for model training. If the database is unavailable during training, the training script falls back to the bundled CSV so that the ML workflow can still be demonstrated. The initial import command requires a working MongoDB server.

Example:
```python
collection.update_one(
    {"student_id": record["student_id"]},
    {"$set": record},
    upsert=True
)
```

## 5. Data Preprocessing
- The feature columns are attendance, study hours, assignments, previous score, internet quality and course.
- Missing numerical values are filled using the median.
- Missing categorical values are filled using the most frequent category.
- Numerical values are standardised using `StandardScaler`.
- Categorical values are converted to numeric columns using `OneHotEncoder(handle_unknown="ignore")`.
- The data is split into 80% training and 20% testing.
- The classification split is stratified to preserve the pass/fail proportion.

Preprocessing is placed inside scikit-learn pipelines. This ensures that imputers, scaling and encoding are fitted only on the training data, reducing data leakage.

## 6. Model Implementation
Two Random Forest models are used.

### 6.1 Random Forest Regressor
The regressor predicts `final_score` as a continuous value. It uses 250 trees, a maximum depth of 12 and a minimum of two samples per leaf.

### 6.2 Random Forest Classifier
The classifier predicts `passed` (0 or 1). It uses 250 trees, a maximum depth of 12, a minimum of two samples per leaf and balanced class weights.

The trained pipelines are saved in the `models/` directory using Joblib. `src/predict.py` demonstrates how to load the models and predict for a new student record.

## 7. Evaluation Results
The evaluation below was calculated from the bundled synthetic dataset using an 80/20 split and random state 42.

### Regression
| Metric | Result |
|---|---:|
| R² Score | 0.3425 |
| Mean Absolute Error (MAE) | 8.3777 |

R² measures the proportion of target variance explained by the model. MAE is the average absolute difference between predicted and actual scores, measured in score points.

### Classification
| Metric | Result |
|---|---:|
| Accuracy | 0.7917 |
| Precision | 0.6042 |
| Recall | 0.4833 |
| F1 Score | 0.5370 |

Accuracy is the fraction of correct predictions. Precision measures how many predicted passes were actual passes. Recall measures how many actual passes were identified. F1 is the harmonic mean of precision and recall.

Confusion matrix (rows = actual class [Fail, Pass], columns = predicted class [Fail, Pass]):
`[[161, 19], [31, 29]]`

## 8. How to Execute the Project
1. Install Python 3.10 or later and start MongoDB Community Server, or configure an Atlas URI.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and configure the connection string.
4. Run `python src/main.py all` from the project root.
5. Review `reports/evaluation_metrics.json`.
6. Run `python src/predict.py` to test a sample prediction.

## 9. Conclusion
This project demonstrates the integration of MongoDB, PyMongo and machine learning in a reproducible Python application. MongoDB stores flexible student documents, PyMongo handles database operations, and Random Forest models perform both score regression and pass/fail classification. The requested evaluation metrics are reported for the test set. Because the dataset is synthetic, the results are suitable for demonstrating the workflow but are not proof of real-world predictive performance. A future version could use an authorised real dataset, compare additional algorithms, tune hyperparameters and perform fairness and privacy assessments.


## 10. Streamlit User Interface
The project includes a Streamlit dashboard (`app.py`) to make the trained models accessible through a browser. The interface provides an overview, a student prediction form, a searchable student dataset table, and a model evaluation page. The prediction form accepts attendance, study hours, assignment completion, previous score, internet quality and course, then displays the predicted final score and pass/fail class. The dashboard reads records from MongoDB when available and otherwise uses the bundled CSV for display. Run it from the project root using `streamlit run app.py`.
