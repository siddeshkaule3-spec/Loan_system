"""Streamlit UI. Styling lives in styles.css, HTML in templates.html, ML in loan_model.py."""
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import loan_model as lm

BASE = Path(__file__).parent

st.set_page_config(page_title="Loan Approval System", page_icon="🏦", layout="wide")


# ------------------------------------------------------------ CSS / HTML loaders
def load_css(path="styles.css"):
    st.markdown(f"<style>{(BASE / path).read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


@st.cache_data
def load_templates(path="templates.html"):
    text = (BASE / path).read_text(encoding="utf-8-sig")
    found = re.findall(r'<template\s+id\s*=\s*["\'](\w+)["\']\s*>(.*?)</template\s*>',
                       text, re.S | re.I)
    return {name: body.strip() for name, body in found}


def render(name, **values):
    if name not in TEMPLATES:
        st.error(f"Template '{name}' not found in templates.html. "
                 f"Templates found: {list(TEMPLATES) or 'none (file empty or wrong format)'}")
        st.stop()
    html = TEMPLATES[name].format(**values)
    st.markdown(html, unsafe_allow_html=True)


load_css()
TEMPLATES = load_templates()
render("hero")

# ------------------------------------------------------------ Data
st.sidebar.header("⚙️ Settings")
uploaded = st.sidebar.file_uploader("Upload loan_approval_data.csv", type="csv")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
seed = int(st.sidebar.number_input("Random state", value=42, step=1))


@st.cache_data
def get_data(file):
    return lm.load_data(file)


@st.cache_resource
def get_models(X, y, num_cols, cat_cols, test_size, seed):
    return lm.train_models(X, y, num_cols, cat_cols, test_size, seed)


try:
    raw = get_data(uploaded)
except FileNotFoundError:
    st.warning(f"Place `{lm.DATA_FILE}` next to `app.py` or upload it from the sidebar.")
    st.stop()

if lm.TARGET not in raw.columns:
    st.error(f"Column '{lm.TARGET}' not found in the dataset.")
    st.stop()

df, X, y, y_raw, num_cols, cat_cols, positive = lm.prepare_data(raw)
fitted, results, cms = get_models(X, y, num_cols, cat_cols, test_size, seed)

tab_data, tab_eda, tab_models, tab_predict = st.tabs(
    ["📄 Data", "📊 EDA", "🤖 Model Performance", "🔮 Predict"])

# ------------------------------------------------------------ Tab 1: Data
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    with c1: render("metric_card", label="Rows", value=f"{len(raw):,}")
    with c2: render("metric_card", label="Columns", value=raw.shape[1])
    with c3: render("metric_card", label="Missing values", value=int(raw.isnull().sum().sum()))
    with c4: render("metric_card", label="Duplicates", value=int(raw.duplicated().sum()))

    st.subheader("Preview")
    st.dataframe(raw.head(50), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Missing values per column")
        st.dataframe(raw.isnull().sum().rename("nulls"), use_container_width=True)
    with right:
        st.subheader("Summary statistics")
        st.dataframe(raw.describe(), use_container_width=True)

# ------------------------------------------------------------ Tab 2: EDA
with tab_eda:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Loan approval distribution")
        fig, ax = plt.subplots()
        counts = y_raw.value_counts()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%")
        st.pyplot(fig)
    with c2:
        if cat_cols:
            st.subheader("Category counts")
            cat_choice = st.selectbox("Categorical column", cat_cols)
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=cat_choice, hue=lm.TARGET, ax=ax)
            plt.xticks(rotation=30)
            st.pyplot(fig)

    if num_cols:
        default = num_cols.index("Applicant_Income") if "Applicant_Income" in num_cols else 0
        num_choice = st.selectbox("Numerical column", num_cols, index=default)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader(f"Distribution of {num_choice}")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=num_choice, bins=20, kde=True, ax=ax)
            st.pyplot(fig)
        with c4:
            st.subheader(f"{num_choice} vs {lm.TARGET}")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=lm.TARGET, y=num_choice, ax=ax)
            st.pyplot(fig)

        st.subheader("Correlation heatmap")
        corr_df = X[num_cols].copy()
        corr_df[lm.TARGET] = y
        corr = corr_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader(f"Correlation with {lm.TARGET}")
        st.dataframe(corr[lm.TARGET].drop(lm.TARGET).sort_values(ascending=False).to_frame("correlation"),
                     use_container_width=True)

# ------------------------------------------------------------ Tab 3: Models
with tab_models:
    best = results["Accuracy"].idxmax()
    st.markdown(f"#### 🏆 Best model by accuracy: **{best}** ({results.loc[best, 'Accuracy']:.2%})")
    st.dataframe(results.style.format("{:.4f}").highlight_max(axis=0, color="#c8f7d4"),
                 use_container_width=True)

    st.subheader("Confusion matrices")
    for col, (name, cm) in zip(st.columns(len(cms)), cms.items()):
        with col:
            fig, ax = plt.subplots(figsize=(3.6, 3.2))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

# ------------------------------------------------------------ Tab 4: Predict
with tab_predict:
    st.subheader("Enter applicant details")
    names = list(fitted.keys())
    model_name = st.selectbox("Model", names, index=names.index(results["Accuracy"].idxmax()))

    with st.form("loan_form"):
        inputs = {}
        cols = st.columns(3)
        for i, col in enumerate(X.columns):
            with cols[i % 3]:
                label = col.replace("_", " ")
                if col in num_cols:
                    s = X[col].dropna()
                    inputs[col] = st.number_input(label, min_value=float(s.min()),
                                                  max_value=float(s.max()) * 2,
                                                  value=float(s.median()))
                else:
                    inputs[col] = st.selectbox(label, sorted(X[col].dropna().astype(str).unique()))
        submitted = st.form_submit_button("Check loan eligibility")

    if submitted:
        label, prob = lm.predict_one(fitted[model_name], inputs, X.columns)
        approved = label == 1
        render("result",
               css_class="approved" if approved else "rejected",
               title="✅ Loan Approved" if approved else "❌ Loan Rejected",
               message=("This application looks eligible for approval." if approved
                        else "This application is unlikely to be approved."),
               probability=f"{prob:.1%}",
               model=model_name,
               bar_style=f"width:{prob * 100:.0f}%")