# 📡 P8 — Customer Churn Analysis & Prediction
**M3 · ML Engine Portfolio · Project 8 of 12**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Source-IBM_Telco-0052CC)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📌 Project Overview

End-to-end churn analysis and prediction on **7,043 IBM Telco customers** — identifying customers likely to leave, understanding why they churn, and providing actionable retention recommendations.

**Core Questions:**
- Which contract and service factors drive churn most?
- Do month-to-month customers pay significantly different charges? (A/B Test)
- How does churn rate evolve across customer tenure cohorts?
- Can ML models reliably predict which customers will churn?

---

## ⚠️ Class Imbalance

| Class | Count | Rate |
|-------|-------|------|
| Retained | 5,174 | 73.5% |
| Churned | 1,869 | 26.5% |

`class_weight='balanced'` applied to all classifiers. Evaluate with **F1, Recall, ROC-AUC**.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | IBM Telco Customer Churn (Kaggle) |
| Records | 7,043 customers |
| Original Features | 33 |
| After Cleaning | 20+ |
| Dropped | 13 columns (IDs, geographic, leakage) |

### Key Cleaning Steps
- Dropped: CustomerID, Count, Country, State, City (ZIP), Lat/Long, Churn Label, Churn Score, CLTV, Churn Reason
- `Total Charges` converted from string to float
- `Churn Score` & `CLTV` dropped — post-churn leakage

---

## 🗂 Project Structure

```
📁 Repo_8_Customer_Churn/
├── Home.py
├── M3_logo.png
├── requirements.txt
├── README.md
├── data/
│   └── churn_clean.csv          ← from P8_clean_data.py
└── pages/
    ├── EDA_dashboard.py         ← 13-tab analysis
    └── ML_Models.py             ← 5-tab ML engine
```

---

## 📈 EDA Dashboard — 13 Tabs

| # | Tab | Highlight |
|---|-----|-----------|
| 1 | Data Overview | Dictionary, stats, class balance |
| 2 | Churn Distribution ★ | By contract, internet service, payment method |
| 3 | Tenure Analysis ★ | Cohort bands + Tenure × Contract heatmap |
| 4 | Charges Analysis ★ | Monthly/Total · scatter tenure vs charges |
| 5 | Service Analysis ★ | Bundle size · churn by individual service |
| 6 | Demographics | Senior · Partner · Dependents · at-risk profile |
| 7 | Feature Engineering | Engineered flags + distributions |
| 8 | Correlation | Full heatmap + top churn predictors |
| 9 | KDE by Segment ★ | Monthly Charges & Tenure KDE: churned vs retained |
| 10 | A/B Test ★ | Month-to-month vs long-term — Welch T-test + Cohen's d |
| 11 | Missing Values | Cleaning strategy |
| 12 | Multicollinearity | VIF analysis |
| 13 | Insights & Report | Findings + recommendations + download |

---

## 🤖 ML Models — 5 Tabs

| Tab | Content |
|-----|---------|
| 1 | Training — 6 Reg + 6 Clf · individual buttons |
| 2 | Regression Results — R², MAE, RMSE · predict Monthly Charges |
| 3 | Classification Results — F1, Precision, Recall, ROC-AUC · confusion matrix |
| 4 | Feature Importance — top churn predictors |
| 5 | Interactive Predict — churn risk scoring with risk factor breakdown |

**Regression (6):** Linear · Ridge · Lasso · Decision Tree · Random Forest · Gradient Boosting

**Classification (6):** Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · SVM (Linear) · KNN

---

## 🔑 Key Findings

**1. Contract Type is #1 Driver**
Month-to-month customers churn at 3–4× the rate of annual contract customers. No switching cost = no loyalty.

**2. First 12 Months are Critical**
Churn rate drops dramatically after year one. The onboarding experience determines long-term retention.

**3. Price Sensitivity**
Churned customers pay significantly higher monthly charges on average. High-charge customers need proactive outreach.

**4. Fiber Optic Paradox**
Fiber optic users churn more than DSL users — higher speed expectations not matched by service quality.

**5. Service Bundle Effect**
Each additional service subscribed reduces churn probability. Bundling is the strongest retention lever after contracts.

---

## 💡 Recommendations

| Priority | Action |
|----------|--------|
| 📝 High | Offer discount to convert month-to-month → annual contract |
| 🎯 High | Intensive first-year loyalty programme (months 1–12) |
| 💰 High | Proactive discount for customers paying > $80/month |
| 🛡 Medium | Subsidise Tech Support + Online Security — halves churn |
| 👴 Medium | Senior citizen dedicated support line |
| 💳 Medium | Incentivise switch from Electronic Check to auto-pay |

---

## 🚀 How to Run

```bash
git clone https://github.com/YourUsername/Repo_8_Customer_Churn.git
cd Repo_8_Customer_Churn
pip install -r requirements.txt

# Step 1: Generate clean dataset
# Run P8_clean_data.py in Jupyter → churn_clean.csv → place in data/

# Step 2: Launch app
streamlit run Home.py
```

---

## 🛠 Tech Stack

`Python 3.11` · `Streamlit` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Scikit-learn` · `SciPy` · `Statsmodels` · `Psutil`

---

**Mohamed · M3 · ML Engine Portfolio — 12 End-to-End Data Science Projects**
