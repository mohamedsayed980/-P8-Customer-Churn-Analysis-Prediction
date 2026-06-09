"""
Repo_8_Customer_Churn — EDA_dashboard.py  (13 Tabs)
Author : Mohamed · M3
Dataset: IBM Telco Customer Churn · 7,043 customers
"""

import pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

warnings.filterwarnings("ignore")
S = st.session_state

st.set_page_config(page_title="EDA · Customer Churn · M3",
                   page_icon="📡", layout="wide")

LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"
DATA = pathlib.Path(__file__).parent.parent / "data" / "churn_clean.csv"

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 📡 EDA Dashboard")
    st.markdown("Customer Churn · 13 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    _uploaded = st.file_uploader("Upload Clean CSV", type=["csv"],
                                  help="Upload churn_clean.csv from Jupyter.",
                                  key="eda_upload")
    if _uploaded is not None:
        st.success(f"✅ Using: {_uploaded.name}")
    else:
        st.info("Using default: churn_clean.csv")

CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","accent":"#00695c",
       "secondary":"#455a64","light":"#e3f2fd","dark":"#1a237e",
       "purple":"#6a1b9a","amber":"#f57f17","pink":"#ad1457",
       "indigo":"#283593","cyan":"#00838f","lime":"#558b2f",
       "brown":"#4e342e","grey":"#546e7a","white":"#ffffff","black":"#212121"}

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{background:#1a2633;border:1.5px dashed #4a7fa5;border-radius:8px;padding:6px;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] *{color:#e0e8f0 !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:#1a2633 !important;border:none !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *{color:#a0bcd4 !important;}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{background:#1565c0 !important;color:#ffffff !important;border:none !important;border-radius:6px !important;}
.main{background:#f4f7fb;}
div[data-testid="metric-container"]{background:#e3f2fd;border-left:4px solid #1565c0;border-radius:6px;padding:10px 14px;}
.sec-header{background:linear-gradient(90deg,#1565c0,#00695c);color:#ffffff !important;
  padding:10px 18px;border-radius:8px;font-size:1.1rem;font-weight:700;margin-bottom:16px;}
.insight-box{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.insight-box p{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}
.warn-box{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.warn-box p{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}
.info-box{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.info-box p{color:#0d2a4a !important;margin:0;font-size:0.93rem;line-height:1.6;}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)

# ── LOAD ─────────────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes=None) -> pd.DataFrame:
    import io as _io
    if file_bytes is not None:
        df = pd.read_csv(_io.BytesIO(file_bytes), sep=",", decimal=".")
    else:
        df = pd.read_csv(DATA, sep=",", decimal=".")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()
    return df

_up = S.get("eda_upload", None)
if _up is not None:
    _bytes = _up.read(); _up.seek(0)
    df = load_data(file_bytes=_bytes)
else:
    df = load_data()

if df.empty:
    st.warning("⚠️ No data. Run P8_clean_data.py first then upload churn_clean.csv.")
    st.stop()

S["df_work"] = df

TARGET  = "Churn Value"
REG_T   = "Monthly Charges"
df_churn  = df[df[TARGET] == 1]
df_retain = df[df[TARGET] == 0]
churn_rate = df[TARGET].mean() * 100

NUM_COLS = [c for c in ["Tenure Months","Monthly Charges","Total Charges",
                         "Num_Services","Charge_per_Month"] if c in df.columns]
CAT_COLS = [c for c in ["Gender","Senior Citizen","Partner","Dependents",
                         "Contract","Internet Service","Payment Method"] if c in df.columns]
ENG_COLS = [c for c in ["high_value","at_risk_profile","is_monthly"] if c in df.columns]
ENC_COLS = [c for c in df.columns if c.endswith("_enc")]

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([
    "1 · Data Overview",
    "2 · Churn Distribution ★",
    "3 · Tenure Analysis ★",
    "4 · Charges Analysis ★",
    "5 · Service Analysis ★",
    "6 · Demographics",
    "7 · Feature Engineering",
    "8 · Correlation",
    "9 · KDE by Segment ★",
    "10 · A/B Test ★",
    "11 · Missing Values",
    "12 · Multicollinearity",
    "13 · Insights & Report",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("📋 Tab 1 — Data Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churned",         f"{len(df_churn):,}")
    c3.metric("Retained",        f"{len(df_retain):,}")
    c4.metric("Churn Rate",      f"{churn_rate:.1f}%")
    c5.metric("Features",        f"{df.shape[1]}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📄 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        sec("📐 Column Info")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  df.dtypes.astype(str).values,
            "Nulls":  df.isnull().sum().values,
        })
        st.dataframe(info_df, use_container_width=True)

    st.markdown("---")
    sec("📊 Descriptive Statistics")
    st.dataframe(df[NUM_COLS].describe().round(3), use_container_width=True)

    st.markdown("---")
    sec("🗂 Data Dictionary")
    dd = pd.DataFrame({
        "Column": ["Tenure Months","Monthly Charges","Total Charges",
                   "Contract","Internet Service","Payment Method",
                   "Phone Service","Multiple Lines","Online Security",
                   "Online Backup","Device Protection","Tech Support",
                   "Streaming TV","Streaming Movies","Paperless Billing",
                   "Gender","Senior Citizen","Partner","Dependents",
                   "Churn Value","Tenure_Band","Num_Services",
                   "high_value","at_risk_profile","is_monthly","Charge_per_Month"],
        "Type": ["Numeric","Numeric","Numeric",
                 "Categorical","Categorical","Categorical",
                 "Binary","Categorical","Categorical",
                 "Categorical","Categorical","Categorical",
                 "Categorical","Categorical","Binary",
                 "Binary","Binary","Binary","Binary",
                 "Target","Engineered","Engineered",
                 "Engineered","Engineered","Engineered","Engineered"],
        "Description": [
            "Months customer has been with company",
            "Monthly charge in USD — regression target",
            "Total amount charged — converted from string",
            "Month-to-month / One year / Two year",
            "DSL / Fiber optic / No",
            "Electronic check / Mailed check / Bank transfer / Credit card",
            "1=Yes 0=No","No/Yes/No phone service",
            "No/Yes/No internet service","No/Yes/No internet service",
            "No/Yes/No internet service","No/Yes/No internet service",
            "No/Yes/No internet service","No/Yes/No internet service",
            "1=Yes 0=No","0=Female 1=Male encoded",
            "1=Yes 0=No","1=Yes 0=No","1=Yes 0=No",
            "1=Churned 0=Retained — classification target",
            "Tenure bucket: 0-12m / 13-24m / 25-48m / 49-72m",
            "Count of services subscribed (0–8)",
            "1 if Monthly Charges > median",
            "1 if Senior + no Partner + no Dependents",
            "1 if Contract = Month-to-month",
            "Total Charges / Tenure (average monthly spend)",
        ]
    })
    st.dataframe(dd, use_container_width=True)
    warn(f"Churn rate {churn_rate:.1f}% — moderate imbalance → class_weight='balanced' in all classifiers.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — CHURN DISTRIBUTION ★
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("🎯 Tab 2 — Churn Distribution ★")
    info("Where does churn concentrate? Contract type is the single most important driver.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Overall Balance")
        bal = pd.DataFrame({"Label":["Retained","Churned"],"Count":[len(df_retain),len(df_churn)]})
        bal["Pct"] = (bal["Count"]/len(df)*100).round(1)
        fig = px.bar(bal, x="Label", y="Count", color="Label",
                     color_discrete_map={"Retained":CLR["primary"],"Churned":CLR["danger"]},
                     text=bal["Pct"].apply(lambda x: f"{x}%"),
                     title="Churned vs Retained")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=370, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("🥧 Churn Proportion")
        fig2 = px.pie(bal, names="Label", values="Count", hole=0.45,
                      color="Label",
                      color_discrete_map={"Retained":CLR["primary"],"Churned":CLR["danger"]},
                      title="Class Distribution")
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(height=370)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Churn Rate by Contract Type")
        ct = df.groupby("Contract")[TARGET].agg(Total="count",Churn="sum").reset_index()
        ct["Churn_Rate%"] = (ct["Churn"]/ct["Total"]*100).round(2)
        ct = ct.sort_values("Churn_Rate%", ascending=False)
        fig3 = px.bar(ct, x="Contract", y="Churn_Rate%",
                      color="Churn_Rate%",
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Churn Rate % by Contract",
                      text=ct["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig3.add_hline(y=churn_rate, line_dash="dash", line_color="blue",
                       annotation_text=f"Avg {churn_rate:.1f}%")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=370)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        sec("📊 Churn Rate by Internet Service")
        it = df.groupby("Internet Service")[TARGET].agg(Total="count",Churn="sum").reset_index()
        it["Churn_Rate%"] = (it["Churn"]/it["Total"]*100).round(2)
        it = it.sort_values("Churn_Rate%", ascending=False)
        fig4 = px.bar(it, x="Internet Service", y="Churn_Rate%",
                      color="Churn_Rate%",
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Churn Rate % by Internet Service",
                      text=it["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig4.add_hline(y=churn_rate, line_dash="dash", line_color="blue",
                       annotation_text=f"Avg {churn_rate:.1f}%")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(height=370)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    sec("📊 Churn Rate by Payment Method")
    pm = df.groupby("Payment Method")[TARGET].agg(Total="count",Churn="sum").reset_index()
    pm["Churn_Rate%"] = (pm["Churn"]/pm["Total"]*100).round(2)
    pm = pm.sort_values("Churn_Rate%", ascending=False)
    fig5 = px.bar(pm, x="Payment Method", y="Churn_Rate%",
                  color="Churn_Rate%",
                  color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                  title="Churn Rate % by Payment Method",
                  text=pm["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
    fig5.add_hline(y=churn_rate, line_dash="dash", line_color="blue",
                   annotation_text=f"Avg {churn_rate:.1f}%")
    fig5.update_traces(textposition="outside")
    fig5.update_layout(height=370)
    st.plotly_chart(fig5, use_container_width=True)

    insight("Month-to-month contracts show dramatically higher churn — locking customers into longer contracts is the #1 retention lever.")
    insight("Fiber optic users churn more — possibly due to higher charges or unmet quality expectations.")
    warn("Electronic check payment = highest churn rate — these customers may lack loyalty program engagement.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — TENURE ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("⏳ Tab 3 — Tenure Analysis ★")
    info("Tenure is the strongest continuous predictor — new customers churn at much higher rates.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Tenure Distribution — Churned vs Retained")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(df_retain["Tenure Months"], bins=30, alpha=0.6,
                color=CLR["primary"], label="Retained", density=True)
        ax.hist(df_churn["Tenure Months"], bins=30, alpha=0.7,
                color=CLR["danger"], label="Churned", density=True)
        ax.axvline(df_retain["Tenure Months"].mean(), color=CLR["primary"],
                   lw=2, ls="--", label=f"Retained mean={df_retain['Tenure Months'].mean():.0f}m")
        ax.axvline(df_churn["Tenure Months"].mean(), color=CLR["danger"],
                   lw=2, ls="--", label=f"Churned mean={df_churn['Tenure Months'].mean():.0f}m")
        ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Density")
        ax.set_title("Tenure: Churned vs Retained"); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Churn Rate by Tenure Band ★")
        if "Tenure_Band" in df.columns:
            tb = df.groupby("Tenure_Band", observed=True)[TARGET].agg(
                Total="count", Churn="sum").reset_index()
            tb["Churn_Rate%"] = (tb["Churn"]/tb["Total"]*100).round(2)
            fig2 = px.bar(tb, x="Tenure_Band", y="Churn_Rate%",
                          color="Churn_Rate%",
                          color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                          title="Churn Rate % by Tenure Band",
                          text=tb["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig2.add_hline(y=churn_rate, line_dash="dash", line_color="blue",
                           annotation_text=f"Avg {churn_rate:.1f}%")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Churned — Avg Tenure",  f"{df_churn['Tenure Months'].mean():.0f} months")
    c2.metric("Retained — Avg Tenure", f"{df_retain['Tenure Months'].mean():.0f} months")
    c3.metric("Churned — Median",      f"{df_churn['Tenure Months'].median():.0f} months")
    c4.metric("Retained — Median",     f"{df_retain['Tenure Months'].median():.0f} months")

    st.markdown("---")
    sec("📊 Tenure Band × Contract — Churn Heatmap")
    if "Tenure_Band" in df.columns:
        heat = df.groupby(["Tenure_Band","Contract"], observed=True)[TARGET].mean().unstack() * 100
        fig3, ax = plt.subplots(figsize=(9,4))
        sns.heatmap(heat.round(1), annot=True, fmt=".1f", cmap="RdYlGn_r",
                    ax=ax, linewidths=0.5, annot_kws={"size":10})
        ax.set_title("Churn Rate % — Tenure Band × Contract Type", fontsize=12)
        ax.set_xlabel("Contract"); ax.set_ylabel("Tenure Band")
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Customers in first 12 months churn at the highest rate — the critical retention window.")
    insight("After 48 months, churn drops dramatically — long-tenure customers are far more loyal.")
    warn("Month-to-month + 0-12m tenure = highest risk combination. Target these customers first.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — CHARGES ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("💰 Tab 4 — Charges Analysis ★")
    info("Higher monthly charges correlate with higher churn — price sensitivity is a key driver.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Churned — Avg Monthly",  f"${df_churn['Monthly Charges'].mean():.2f}")
    c2.metric("Retained — Avg Monthly", f"${df_retain['Monthly Charges'].mean():.2f}")
    c3.metric("Churned — Avg Total",    f"${df_churn['Total Charges'].mean():,.0f}")
    c4.metric("Retained — Avg Total",   f"${df_retain['Total Charges'].mean():,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Monthly Charges — Churned vs Retained")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(df_retain["Monthly Charges"], bins=40, alpha=0.6,
                color=CLR["primary"], label="Retained", density=True)
        ax.hist(df_churn["Monthly Charges"], bins=40, alpha=0.7,
                color=CLR["danger"], label="Churned", density=True)
        ax.axvline(df_retain["Monthly Charges"].mean(), color=CLR["primary"], lw=2, ls="--")
        ax.axvline(df_churn["Monthly Charges"].mean(),  color=CLR["danger"],  lw=2, ls="--")
        ax.set_xlabel("Monthly Charges ($)"); ax.set_ylabel("Density")
        ax.set_title("Monthly Charges: Churned vs Retained"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📦 Box Plot — Monthly Charges by Churn")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        bp = ax2.boxplot([df_retain["Monthly Charges"].dropna(),
                          df_churn["Monthly Charges"].dropna()],
                         patch_artist=True, labels=["Retained","Churned"])
        bp["boxes"][0].set_facecolor(CLR["light"])
        bp["boxes"][1].set_facecolor("#fce4ec")
        for m in bp["medians"]: m.set_color(CLR["danger"]); m.set_linewidth(2)
        ax2.set_ylabel("Monthly Charges ($)")
        ax2.set_title("Monthly Charges by Churn Status")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Monthly Charges vs Tenure — Scatter by Churn")
    fig3 = px.scatter(df.sample(min(2000, len(df)), random_state=42),
                      x="Tenure Months", y="Monthly Charges",
                      color=TARGET, opacity=0.5,
                      color_discrete_map={0:CLR["primary"], 1:CLR["danger"]},
                      labels={TARGET:"Churn"},
                      title="Monthly Charges vs Tenure — Churn Highlighted")
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)

    insight("Churned customers pay significantly higher monthly charges on average.")
    insight("Low tenure + high monthly charge = highest churn risk zone — visible in scatter plot.")
    warn("Total Charges is low for churned customers — they leave before accumulating long billing history.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — SERVICE ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("📦 Tab 5 — Service Analysis ★")
    info("Customers with more services churn less — service bundling is a powerful retention tool.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Churn Rate by Number of Services ★")
        if "Num_Services" in df.columns:
            ns = df.groupby("Num_Services")[TARGET].agg(Total="count",Churn="sum").reset_index()
            ns["Churn_Rate%"] = (ns["Churn"]/ns["Total"]*100).round(2)
            fig = px.bar(ns, x="Num_Services", y="Churn_Rate%",
                         color="Churn_Rate%",
                         color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                         title="Churn Rate % by Number of Services",
                         text=ns["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig.add_hline(y=churn_rate, line_dash="dash", line_color="blue",
                          annotation_text=f"Avg {churn_rate:.1f}%")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380, xaxis_title="Number of Services")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Service Count — Churned vs Retained")
        if "Num_Services" in df.columns:
            fig2, ax = plt.subplots(figsize=(7,4))
            ax.hist(df_retain["Num_Services"], bins=9, alpha=0.6,
                    color=CLR["primary"], label="Retained", density=True)
            ax.hist(df_churn["Num_Services"], bins=9, alpha=0.7,
                    color=CLR["danger"], label="Churned", density=True)
            ax.set_xlabel("Number of Services"); ax.set_ylabel("Density")
            ax.set_title("Service Count: Churned vs Retained"); ax.legend()
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Churn Rate by Individual Service")
    service_cols_orig = ["Online Security","Online Backup","Device Protection",
                         "Tech Support","Streaming TV","Streaming Movies"]
    service_cols_orig = [c for c in service_cols_orig if c in df.columns]
    svc_rows = []
    for col in service_cols_orig:
        for val in df[col].unique():
            sub = df[df[col]==val]
            cr  = sub[TARGET].mean()*100
            svc_rows.append({"Service":col,"Value":val,
                              "Count":len(sub),"Churn_Rate%":round(cr,2)})
    svc_df = pd.DataFrame(svc_rows)
    # Show "Yes" vs "No" only
    svc_yes = svc_df[svc_df["Value"]=="Yes"].sort_values("Churn_Rate%",ascending=False)
    svc_no  = svc_df[svc_df["Value"]=="No"].sort_values("Churn_Rate%",ascending=False)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.bar(svc_yes, x="Service", y="Churn_Rate%",
                      color="Churn_Rate%",
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Churn Rate — Customers WITH Each Service",
                      text=svc_yes["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=380, xaxis_tickangle=-20)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.bar(svc_no, x="Service", y="Churn_Rate%",
                      color="Churn_Rate%",
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Churn Rate — Customers WITHOUT Each Service",
                      text=svc_no["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig4.update_traces(textposition="outside")
        fig4.update_layout(height=380, xaxis_tickangle=-20)
        st.plotly_chart(fig4, use_container_width=True)

    insight("Customers without Tech Support or Online Security churn at much higher rates.")
    insight("Each additional service subscribed reduces churn probability — bundle strategy works.")
    warn("Streaming services show weaker retention effect — content-only customers are less sticky.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("👥 Tab 6 — Demographics")

    demo_cols = [c for c in ["Gender","Senior Citizen","Partner","Dependents"] if c in df.columns]
    cols = st.columns(2)
    for i, col_name in enumerate(demo_cols):
        with cols[i % 2]:
            sec(f"📊 Churn Rate by {col_name}")
            dc = df.groupby(col_name)[TARGET].agg(Total="count",Churn="sum").reset_index()
            dc["Churn_Rate%"] = (dc["Churn"]/dc["Total"]*100).round(2)
            fig = px.bar(dc, x=col_name, y="Churn_Rate%",
                         color="Churn_Rate%",
                         color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                         title=f"Churn Rate % by {col_name}",
                         text=dc["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig.add_hline(y=churn_rate, line_dash="dash", line_color="blue",
                          annotation_text=f"Avg {churn_rate:.1f}%")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    sec("📊 At-Risk Profile — Senior + No Partner + No Dependents")
    if "at_risk_profile" in df.columns:
        ar = df.groupby("at_risk_profile")[TARGET].agg(Total="count",Churn="sum").reset_index()
        ar["Label"]       = ar["at_risk_profile"].map({0:"Standard",1:"At-Risk Profile"})
        ar["Churn_Rate%"] = (ar["Churn"]/ar["Total"]*100).round(2)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(ar[["Label","Total","Churn","Churn_Rate%"]], use_container_width=True)
        with col2:
            fig2 = px.bar(ar, x="Label", y="Churn_Rate%", color="Label",
                          color_discrete_map={"Standard":CLR["success"],
                                              "At-Risk Profile":CLR["danger"]},
                          title="At-Risk Profile vs Standard",
                          text=ar["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    insight("Senior citizens without partner or dependents show highest churn — isolation drives switching.")
    warn("Gender shows minimal churn difference — gender-based targeting not effective.")

# ══════════════════════════════════════════════════════════════
# TAB 7 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    sec("⚙️ Tab 7 — Feature Engineering")

    fe = pd.DataFrame({
        "Feature":     ["Tenure_Band","Num_Services","high_value",
                        "at_risk_profile","is_monthly","Charge_per_Month",
                        "*_enc columns"],
        "Source":      ["Tenure Months cut into 4 bands",
                        "Count of 8 service columns == 'Yes'",
                        "Monthly Charges > median",
                        "Senior=Yes + Partner=No + Dependents=No",
                        "Contract == 'Month-to-month'",
                        "Total Charges / max(Tenure,1)",
                        "LabelEncoder on all categoricals"],
        "Reason":      ["Non-linear tenure effect — cohort analysis",
                        "Service bundle = retention proxy",
                        "Price sensitivity segmentation",
                        "Vulnerable customer profile",
                        "Highest-risk contract flag",
                        "Normalised spend rate per month",
                        "ML-ready categorical features"],
    })
    st.dataframe(fe, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 is_monthly Flag — Churn Rate")
        if "is_monthly" in df.columns:
            im = df.groupby("is_monthly")[TARGET].agg(Total="count",Churn="sum").reset_index()
            im["Label"]       = im["is_monthly"].map({0:"Long-Term",1:"Month-to-Month"})
            im["Churn_Rate%"] = (im["Churn"]/im["Total"]*100).round(2)
            fig = px.bar(im, x="Label", y="Churn_Rate%", color="Label",
                         color_discrete_map={"Long-Term":CLR["success"],
                                             "Month-to-Month":CLR["danger"]},
                         title="Churn Rate: Monthly vs Long-Term",
                         text=im["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 high_value Flag — Churn Rate")
        if "high_value" in df.columns:
            hv = df.groupby("high_value")[TARGET].agg(Total="count",Churn="sum").reset_index()
            hv["Label"]       = hv["high_value"].map({0:"Standard Value",1:"High Value"})
            hv["Churn_Rate%"] = (hv["Churn"]/hv["Total"]*100).round(2)
            fig2 = px.bar(hv, x="Label", y="Churn_Rate%", color="Label",
                          color_discrete_map={"Standard Value":CLR["success"],
                                              "High Value":CLR["danger"]},
                          title="Churn Rate: High Value vs Standard",
                          text=hv["Churn_Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    insight("is_monthly flag captures the contract risk in one binary — highly predictive for ML.")
    info("_enc columns used for ML only — original columns kept for EDA display.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — CORRELATION
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("🔥 Tab 8 — Correlation Analysis")

    corr_cols = [c for c in NUM_COLS + ENG_COLS + ENC_COLS + [TARGET] if c in df.columns]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13,9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size":8})
    ax.set_title("Correlation Matrix — Customer Churn Features", fontsize=13, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    sec("🎯 Top Correlations with Churn Value")
    tgt = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10,5))
    colors_bar = [CLR["danger"] if v>0 else CLR["success"] for v in tgt.values]
    ax2.barh(tgt.index, tgt.values, color=colors_bar)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Pearson r with Churn Value")
    ax2.set_title("Feature Correlation with Churn", fontsize=12, fontweight="bold")
    for i,(idx,val) in enumerate(tgt.items()):
        ax2.text(val+0.005 if val>=0 else val-0.005, i,
                 f"{val:.3f}", va="center",
                 ha="left" if val>=0 else "right", fontsize=9)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("Tenure Months is negatively correlated with churn — longer tenure = lower churn.")
    insight("is_monthly and Month-to-month contract_enc show strong positive correlation with churn.")
    warn("Pearson r misses non-linear effects — use RF/GB feature importance for full picture.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — KDE BY SEGMENT ★
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("📈 Tab 9 — KDE by Segment ★")
    info("KDE (Kernel Density Estimation) shows the full distribution shape — not just mean/median. "
         "Reveals where churned and retained customers differ most.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Monthly Charges KDE — Churned vs Retained")
        fig, ax = plt.subplots(figsize=(8,5))
        df_churn["Monthly Charges"].plot.kde(ax=ax, color=CLR["danger"],
                                              lw=2.5, label="Churned")
        df_retain["Monthly Charges"].plot.kde(ax=ax, color=CLR["primary"],
                                               lw=2.5, label="Retained")
        ax.fill_between(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(),
                        alpha=0.15, color=CLR["danger"])
        ax.fill_between(ax.lines[1].get_xdata(), ax.lines[1].get_ydata(),
                        alpha=0.10, color=CLR["primary"])
        ax.set_xlabel("Monthly Charges ($)"); ax.set_ylabel("Density")
        ax.set_title("KDE: Monthly Charges by Churn Status")
        ax.legend(); ax.set_xlim(0, None)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Tenure Months KDE — Churned vs Retained")
        fig2, ax2 = plt.subplots(figsize=(8,5))
        df_churn["Tenure Months"].plot.kde(ax=ax2, color=CLR["danger"],
                                            lw=2.5, label="Churned")
        df_retain["Tenure Months"].plot.kde(ax=ax2, color=CLR["primary"],
                                             lw=2.5, label="Retained")
        ax2.fill_between(ax2.lines[0].get_xdata(), ax2.lines[0].get_ydata(),
                         alpha=0.15, color=CLR["danger"])
        ax2.fill_between(ax2.lines[1].get_xdata(), ax2.lines[1].get_ydata(),
                         alpha=0.10, color=CLR["primary"])
        ax2.set_xlabel("Tenure (months)"); ax2.set_ylabel("Density")
        ax2.set_title("KDE: Tenure by Churn Status")
        ax2.legend(); ax2.set_xlim(0, None)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 KDE by Contract Segment — Monthly Charges")
    fig3, ax3 = plt.subplots(figsize=(12,5))
    contract_colors = [CLR["danger"], CLR["warning"], CLR["success"]]
    for i, ct in enumerate(df["Contract"].unique()):
        sub = df[df["Contract"]==ct]["Monthly Charges"]
        sub.plot.kde(ax=ax3, lw=2, label=ct, color=contract_colors[i % 3])
    ax3.set_xlabel("Monthly Charges ($)"); ax3.set_ylabel("Density")
    ax3.set_title("KDE: Monthly Charges by Contract Type")
    ax3.legend(); ax3.set_xlim(0, None)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Churned customers cluster at HIGH monthly charges — price sensitivity is clear in KDE shape.")
    insight("Retained customers show a bimodal distribution — low-cost basics OR full-bundle users stay.")
    insight("Month-to-month KDE is wider and right-skewed — more price-variable and churn-prone.")

# ══════════════════════════════════════════════════════════════
# TAB 10 — A/B TEST ★
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    sec("🧪 Tab 10 — A/B Test: Month-to-Month vs Long-Term ★")
    info("Hypothesis: Month-to-month customers pay significantly different monthly charges "
         "than long-term contract customers.")

    if "is_monthly" in df.columns:
        gA = df[df["is_monthly"]==0]["Monthly Charges"]
        gB = df[df["is_monthly"]==1]["Monthly Charges"]

        t_stat, p_val = stats.ttest_ind(gA, gB, equal_var=False)
        pooled        = np.sqrt((gA.std()**2 + gB.std()**2) / 2)
        cohens_d      = (gB.mean() - gA.mean()) / pooled
        diff          = gB.mean() - gA.mean()
        se            = np.sqrt(gA.var()/len(gA) + gB.var()/len(gB))
        ci_lo, ci_hi  = diff - 1.96*se, diff + 1.96*se

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Long-Term — Avg Monthly",    f"${gA.mean():.2f}")
        c2.metric("Month-to-Month — Avg Monthly",f"${gB.mean():.2f}")
        c3.metric("p-value",                     f"{p_val:.6f}")
        c4.metric("Cohen's d",                   f"{cohens_d:.3f}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.hist(gA, bins=40, alpha=0.6, color=CLR["success"],
                    label=f"Long-Term (n={len(gA):,})", density=True)
            ax.hist(gB, bins=40, alpha=0.6, color=CLR["danger"],
                    label=f"Month-to-Month (n={len(gB):,})", density=True)
            ax.axvline(gA.mean(), color=CLR["success"], lw=2, ls="--")
            ax.axvline(gB.mean(), color=CLR["danger"],  lw=2, ls="--")
            ax.set_xlabel("Monthly Charges ($)"); ax.legend()
            ax.set_title("Distribution: Contract Groups")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            bp = ax2.boxplot([gA.dropna(), gB.dropna()], patch_artist=True,
                             labels=["Long-Term","Month-to-Month"])
            bp["boxes"][0].set_facecolor(CLR["light"])
            bp["boxes"][1].set_facecolor("#fce4ec")
            for m in bp["medians"]: m.set_color(CLR["danger"]); m.set_linewidth(2)
            ax2.set_ylabel("Monthly Charges ($)")
            ax2.set_title("Box Plot: Contract Groups")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown("---")
        sec("📋 Test Results")
        res = pd.DataFrame({
            "Metric": ["Test","H₀","H₁","t-statistic","p-value",
                       "Significant (α=0.05)","Cohen's d","Effect Size",
                       "95% CI (difference)","Decision"],
            "Result": [
                "Welch T-Test (unequal variance)",
                "Month-to-month charges = Long-term charges",
                "Month-to-month charges ≠ Long-term charges",
                f"{t_stat:.4f}", f"{p_val:.6f}",
                "✅ YES" if p_val < 0.05 else "❌ NO",
                f"{cohens_d:.4f}",
                "Large" if abs(cohens_d)>0.8 else "Medium" if abs(cohens_d)>0.5 else "Small",
                f"[{ci_lo:.2f}, {ci_hi:.2f}]",
                "✅ REJECT H₀" if p_val < 0.05 else "❌ FAIL to reject H₀"
            ]
        })
        st.dataframe(res, use_container_width=True)

        if p_val < 0.05:
            diff_pct = (gB.mean()-gA.mean())/gA.mean()*100
            insight(f"Month-to-month customers pay ${gB.mean():.2f}/mo vs ${gA.mean():.2f}/mo "
                    f"for long-term — {diff_pct:+.1f}% difference. Statistically significant.")
            warn("Offering month-to-month customers a discount to switch to annual contracts "
                 "is a high-ROI retention strategy.")

# ══════════════════════════════════════════════════════════════
# TAB 11 — MISSING VALUES
# ══════════════════════════════════════════════════════════════
with tabs[10]:
    sec("❓ Tab 11 — Missing Values")

    null_counts = df.isnull().sum()
    null_pct    = (null_counts/len(df)*100).round(2)
    miss = pd.DataFrame({
        "Column":  null_counts.index,
        "Missing": null_counts.values,
        "Percent": null_pct.values,
    }).sort_values("Missing", ascending=False).reset_index(drop=True)
    miss_only = miss[miss["Missing"] > 0]

    if miss_only.empty:
        insight("No missing values — dataset is clean after preprocessing.")
    else:
        st.dataframe(miss_only, use_container_width=True)
        fig = px.bar(miss_only, x="Column", y="Percent",
                     color="Percent",
                     color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                     title="Missing % per Column")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    sec("📋 Cleaning Strategy Applied")
    st.dataframe(pd.DataFrame({
        "Column":   ["Churn Reason","Total Charges","CustomerID etc."],
        "Issue":    ["73% missing + post-churn info",
                     "Stored as string (object dtype)",
                     "ID / geographic / leakage columns"],
        "Action":   ["Dropped entirely",
                     "pd.to_numeric(errors='coerce') → median fill",
                     "Dropped — 13 columns removed"]
    }), use_container_width=True)
    info("Total Charges had blanks for customers with 0 tenure — filled with median after conversion.")

# ══════════════════════════════════════════════════════════════
# TAB 12 — MULTICOLLINEARITY
# ══════════════════════════════════════════════════════════════
with tabs[11]:
    sec("🔁 Tab 12 — Multicollinearity / VIF")
    info("VIF > 10 = severe multicollinearity. Critical for Logistic Regression.")

    vif_cols = [c for c in NUM_COLS + ENG_COLS if c in df.columns]
    vif_data = df[vif_cols].dropna()
    try:
        vif_df = pd.DataFrame({
            "Feature": vif_cols,
            "VIF": [round(variance_inflation_factor(vif_data.values, i),2)
                    for i in range(len(vif_cols))]
        }).sort_values("VIF", ascending=False)
        vif_df["Risk"] = vif_df["VIF"].apply(
            lambda v: "🔴 High" if v>10 else "🟡 Medium" if v>5 else "🟢 Low")

        col1, col2 = st.columns([1,1.5])
        with col1:
            st.dataframe(vif_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(7,5))
            colors_vif = [CLR["danger"] if v>10 else CLR["warning"] if v>5
                          else CLR["success"] for v in vif_df["VIF"]]
            ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors_vif)
            ax.axvline(10, color=CLR["danger"],  lw=2, ls="--", label="VIF=10")
            ax.axvline(5,  color=CLR["warning"], lw=1.5, ls=":",  label="VIF=5")
            ax.set_xlabel("VIF"); ax.set_title("VIF — Multicollinearity Check")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()
    except Exception as e:
        warn(f"VIF error: {e}")

    warn("Monthly Charges + Total Charges + Charge_per_Month may show high VIF — expected (derived).")
    insight("Tree-based models (RF, GB) are immune to multicollinearity — only matters for Logistic Regression.")

# ══════════════════════════════════════════════════════════════
# TAB 13 — INSIGHTS & REPORT
# ══════════════════════════════════════════════════════════════
with tabs[12]:
    sec("💡 Tab 13 — Insights & Recommendations")

    st.markdown(f"### 📡 Customer Churn — Final Report")
    st.markdown(f"**{len(df):,} customers · {len(df_churn):,} churned ({churn_rate:.1f}%) · IBM Telco · M3**")
    st.markdown("---")

    sec("1️⃣ Top Churn Drivers")
    insight("Month-to-month contract — by far the strongest predictor. These customers have no switching cost.")
    insight("Short tenure (0–12 months) — the critical window. Most churn happens in year one.")
    insight("High monthly charges — price-sensitive customers leave when bills are high.")
    insight("Fiber optic internet — higher speed but also higher cost and higher churn expectation.")
    insight("No Tech Support / No Online Security — unprotected customers feel less invested.")

    st.markdown("---")
    sec("2️⃣ Retention Recommendations")
    recs = [
        ("📝 Convert Month-to-Month",
         "Offer 1–2 month free discount to switch to annual contract — reduces churn by 3–4×."),
        ("🎯 First-Year Program",
         "Intensive onboarding + loyalty rewards in months 1–12 — the highest-risk window."),
        ("💰 High-Charge Alert",
         "Proactively contact customers whose monthly charges exceed $80 with a loyalty discount."),
        ("🛡 Bundle Security Services",
         "Subsidise Tech Support + Online Security — customers with these churn at half the rate."),
        ("👴 Senior Outreach",
         "Senior citizens without partner/dependents are high-risk — dedicated support line recommended."),
        ("💳 Payment Method Incentive",
         "Reward customers who switch from Electronic Check to auto-pay — lower churn, lower costs."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""CUSTOMER CHURN — FINAL REPORT
M3 · IBM Telco · {len(df):,} Customers
Churned: {len(df_churn):,} ({churn_rate:.1f}%) | Retained: {len(df_retain):,}

TOP CHURN DRIVERS:
1. Month-to-month contract — strongest predictor
2. Short tenure (0–12 months) — critical window
3. High monthly charges — price sensitivity
4. Fiber optic without support services
5. No Tech Support / Online Security

RECOMMENDATIONS:
- Convert monthly contracts → annual (discount offer)
- Intensive first-year loyalty programme
- Alert + discount for high-charge customers (>$80)
- Bundle security services — halves churn rate
- Senior outreach programme (no partner/dependents)
- Incentivise auto-pay switch from electronic check
"""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Report (.txt)", report_txt,
                           file_name="CustomerChurn_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        st.download_button("📥 Download Clean Data (.csv)",
                           df.to_csv(index=False),
                           file_name="churn_clean_M3.csv",
                           mime="text/csv", use_container_width=True)
