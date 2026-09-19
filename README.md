# 🏦 Loan Default Risk Predictor

An end-to-end machine learning pipeline designed to predict credit default risk on highly imbalanced financial data. 

This project bypasses the "accuracy paradox" by optimizing for **ROC-AUC** and tuning decision thresholds to maximize recall. It features a complete relational data aggregation framework and provides legally compliant, feature-level interpretability for automated loan rejections using SHAP.

## 🚀 Business Value & Key Results
* **The Problem:** The dataset features a severe 92:8 class imbalance. A naive model predicting "No Default" for everyone achieves 92% accuracy but costs the bank millions in bad loans.
* **The Solution:** Implemented `XGBClassifier` and `RandomForestClassifier` with balanced class weights to penalize minority class misclassifications.
* **The Metrics:** Achieved a **0.78 ROC-AUC score**.
* **Threshold Tuning:** Shifted the decision boundary from the default `0.50` to a custom business threshold of `0.15`. This intentionally trades precision to drastically increase the **Recall** of actual defaulters, catching high-risk applications that standard models miss.

## 🛠️ Technology Stack
* **Data Engineering:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost` (Pipelines, ColumnTransformer, RandomizedSearchCV)
* **Model Interpretability:** `shap`
* **Deployment:** `FastAPI` (Backend Inference), `Streamlit` (Frontend UI)

## 🏗️ Project Architecture

1. **Relational Data Aggregation:** Merged secondary financial histories (bureau data, previous applications) using custom Pandas grouping functions to engineer high-signal numerical and categorical features.
2. **Preprocessing Pipeline:** Scikit-Learn `ColumnTransformer` handles median imputation, standard scaling, and one-hot encoding natively.
3. **Model Interpretability:** Integrated SHAP (SHapley Additive exPlanations) to explain the "black box."
*(Optional: Add a screenshot of your SHAP summary plot here)*

## 💻 Two-Tier Web Deployment
The inference engine is decoupled from the user interface, mimicking production environments:
* **FastAPI Backend:** Loads the `joblib` pipeline into memory and serves predictions via a `/predict` REST endpoint.
* **Streamlit Frontend:** Provides an interactive UI for non-technical stakeholders to input applicant data and view live risk assessments.
*(Optional: Add a screenshot of your Streamlit UI here)*

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/Sunny-Kumar-1/Loan-Default-Predictor.git](https://github.com/Sunny-Kumar-1/Loan-Default-Predictor.git)
cd Loan-Default-Predictor
```

Install dependencies
```bash
pip install -r requirements.txt
```

