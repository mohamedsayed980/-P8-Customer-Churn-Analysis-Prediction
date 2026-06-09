"""
Repo_8_Customer_Churn — Home.py
Author : Mohamed · M3
"""
# streamlit run "E:\FINAL PROJECTS\P8_customer_churn\Home.py"

import pathlib
import streamlit as st

st.set_page_config(page_title="Customer Churn · M3", page_icon="📡", layout="wide")

LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 📡 Customer Churn")
    st.markdown("M3 · ML Engine · P8")
    st.divider()
    st.markdown("**Navigate:**")
    st.markdown("📊 EDA Dashboard → 13 tabs")
    st.markdown("🤖 ML Models     → 5 tabs")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
.main{background:#f4f7fb;}
.hero{background:linear-gradient(135deg,#1a237e,#00695c);
      padding:48px 40px;border-radius:14px;margin-bottom:28px;}
.hero h1{color:#ffffff !important;font-size:2.4rem;font-weight:800;margin:0 0 8px 0;}
.hero p{color:#b2dfdb !important;font-size:1.08rem;margin:0;}
.card{background:#ffffff;border-radius:10px;padding:22px 24px;
      box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:4px solid #1565c0;}
.card h3{color:#1565c0 !important;margin:0 0 8px 0;font-size:1.05rem;}
.card p{color:#37474f !important;font-size:0.92rem;margin:0;line-height:1.6;}
.stat-card{background:#ffffff;border-radius:10px;padding:18px;text-align:center;
           box-shadow:0 2px 10px rgba(0,0,0,0.07);border-bottom:3px solid #1565c0;}
.stat-num{font-size:1.9rem;font-weight:800;color:#1565c0 !important;}
.stat-lbl{font-size:0.82rem;color:#546e7a !important;margin-top:4px;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📡 Customer Churn Analysis</h1>
  <p>End-to-end ML pipeline · 7,043 customers · IBM Telco Dataset · M3 Portfolio · Project 8 of 12</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
for col, (num, lbl) in zip([c1,c2,c3,c4,c5], [
    ("7,043","Customers"), ("26.5%","Churn Rate"),
    ("20+","Features"), ("13","EDA Tabs"), ("12","ML Models")]):
    col.markdown(f"""<div class="stat-card">
      <div class="stat-num">{num}</div>
      <div class="stat-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📌 About This Project")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="card"><h3>🎯 Objective</h3>
    <p>Identify customers likely to churn using behavioral, contractual,
    and service-usage signals. Reduce churn through targeted retention strategies.</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card"><h3>📊 Dataset</h3>
    <p>IBM Telco · 7,043 customers · 33 original features.
    Engineered: Tenure_Band, Num_Services, at_risk_profile,
    high_value, is_monthly, Charge_per_Month.</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card"><h3>🔑 Key Signals</h3>
    <p>Contract type · Tenure months · Monthly charges ·
    Internet service · Tech support · Payment method ·
    Number of services subscribed.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    st.markdown("### 📈 EDA Dashboard — 13 Tabs")
    for num, name, desc in [
        ("1","Data Overview","Shape, types, stats, dictionary"),
        ("2","Churn Distribution ★","By contract, payment, internet service"),
        ("3","Tenure Analysis ★","Cohort bands · churn rate by tenure"),
        ("4","Charges Analysis ★","Monthly/Total · KDE churned vs retained"),
        ("5","Service Analysis ★","Bundle size · churn by service type"),
        ("6","Demographics","Gender · Senior · Partner · Dependents"),
        ("7","Feature Engineering","Flags + distributions"),
        ("8","Correlation","Heatmap + top churn predictors"),
        ("9","KDE by Segment ★","MonthlyCharges distribution: churned vs retained"),
        ("10","A/B Test ★","Month-to-month vs long-term — Welch + Cohen's d"),
        ("11","Missing Values","Imputation strategy"),
        ("12","Multicollinearity","VIF analysis"),
        ("13","Insights & Report","Findings + recommendations + download"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

with col5:
    st.markdown("### 🤖 ML Models — 5 Tabs")
    for num, name, desc in [
        ("1","Model Training","6 Reg + 6 Clf · individual buttons"),
        ("2","Regression Results","R², MAE, RMSE · predict Monthly Charges"),
        ("3","Classification Results","F1, Precision, Recall, ROC-AUC · predict Churn"),
        ("4","Feature Importance","Top churn predictors"),
        ("5","Predict","Interactive churn risk scoring"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("**Churn Rate: 26.5%** — moderate imbalance.\n\n"
               "All classifiers use `class_weight='balanced'`.\n\n"
               "Evaluate with **F1, Recall, ROC-AUC**.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#90a4ae;font-size:0.85rem;'>"
            "Mohamed · M3 · ML Engine Portfolio · Project 8 of 12 · Customer Churn</p>",
            unsafe_allow_html=True)
