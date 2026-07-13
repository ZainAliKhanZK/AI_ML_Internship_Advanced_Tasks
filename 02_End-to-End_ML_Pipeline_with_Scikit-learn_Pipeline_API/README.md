# 📊 Customer Churn Prediction Pipeline

An end-to-end, production-ready machine learning pipeline that predicts customer churn using Scikit-learn's `Pipeline` API. Combines preprocessing, model training, and hyperparameter tuning into a single reusable, exportable artifact.

## 🎯 Problem Statement

Telecom companies lose significant revenue from customer churn, but identifying at-risk customers manually is impractical at scale. A reliable, reusable prediction pipeline is needed to flag customers likely to churn before they leave.

## 🚀 Goal

Build a production-ready ML pipeline using Scikit-learn's `Pipeline` API to preprocess the Telco Customer Churn dataset (scaling numeric features, encoding categorical ones), train and compare Logistic Regression and Random Forest models, tune hyperparameters with `GridSearchCV`, and export the complete pipeline for reuse.

## 📊 Results

| Model | Accuracy | F1-score (churn class) |
|---|---|---|
| **Logistic Regression** (selected) | 80.6% | 0.60 |
| Random Forest | 80.0% | 0.58 |

Logistic Regression was selected as the final model, slightly outperforming Random Forest on churn-class recall — meaning it caught more actual churners.

## 🧠 Approach

1. **Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customer records with demographic, account, and service usage features.
2. **Preprocessing**: Built a `ColumnTransformer` combining a numeric pipeline (median imputation + `StandardScaler`) and a categorical pipeline (most-frequent imputation + `OneHotEncoder`), wrapped inside a single `Pipeline`.
3. **Modeling**: Trained and compared Logistic Regression and Random Forest classifiers.
4. **Hyperparameter tuning**: Used `GridSearchCV` (5-fold cross-validation, scored on F1) to tune regularization strength for Logistic Regression and tree depth/estimators for Random Forest.
5. **Evaluation**: Assessed models using accuracy, F1-score, confusion matrix, and ROC-AUC — with particular attention to churn-class recall, since missing a real churner is costlier than a false alarm.
6. **Export**: Saved the complete fitted pipeline (preprocessing + model) with `joblib`, enabling predictions directly on raw, unprocessed customer data.

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer, GridSearchCV)
- joblib

## 📁 Project Structure

```
├── churn_pipeline.py               # Preprocessing, training, and tuning script
├── churn_prediction_pipeline.joblib  # Exported final pipeline
├── requirements.txt
└── README.md
```

## 💻 Usage

**Setup:**
```bash
git clone https://github.com/ZainAliKhanZK/churn-prediction-pipeline.git
cd churn-prediction-pipeline
pip install -r requirements.txt
```

**Load the pipeline and predict on new customer data:**
```python
import joblib
import pandas as pd

pipeline = joblib.load("churn_prediction_pipeline.joblib")

new_customer = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 45.0, "TotalCharges": 2700.0,
}])

prediction = pipeline.predict(new_customer)
probability = pipeline.predict_proba(new_customer)[:, 1]

print("Churn" if prediction[0] == 1 else "No Churn", f"({probability[0]*100:.1f}% churn probability)")
```

No manual preprocessing needed — the pipeline handles scaling and encoding internally, as long as the input data follows the same column structure as the original dataset.

## ⚠️ Notes & Limitations

- The pipeline expects the exact same feature columns and categorical value spellings used during training. It is specific to this feature schema and is not a general-purpose churn model for other datasets.
- Both models sit in the 0.58–0.60 F1 range on the churn class, consistent with published benchmarks for this dataset — Telco churn prediction is a genuinely hard, imbalanced classification problem.
- Potential future improvements: class weighting (`class_weight="balanced"`) or SMOTE oversampling to improve churn-class recall further.

## 👤 Author

**Zain Ali Khan**
[GitHub](https://github.com/ZainAliKhanZK) · [Hugging Face](https://huggingface.co/ZainAliKhanZK)
