import streamlit as st
import pandas as pd
import numpy as np
import io

from src.features import (
    load_raw, engineer, ATTRIBUTE_FEATURES, INTERACTION_FEATURES
)
from src.model import build_and_train, explain_prediction, friendly_feature_name

st.set_page_config(
    page_title="Snitch Slow-Mover Predictor",
    page_icon="👕",
    layout="wide",
)

DATA_PATH = "data/snitch_slow_mover_dataset.xlsx"

CATEGORY_OPTIONS = ["Shirts", "T-Shirts", "Jeans", "Trousers", "Co-ords", "Shorts", "Jackets", "Hoodies", "Sweaters"]
FABRIC_OPTIONS = ["Cotton", "Polyester", "Linen", "Denim", "Fleece", "Blended"]
DESIGN_OPTIONS = ["Solid", "Printed", "Striped", "Checked", "Textured"]
FIT_OPTIONS = ["Slim", "Regular", "Oversized", "Relaxed"]
ASP_OPTIONS = ["699-999", "1000-1299", "1300-1599", "1600-1999"]
SLEEVE_OPTIONS = ["Full Sleeve", "Half Sleeve", "Sleeveless", "Not Applicable"]
SIZE_AVAIL_OPTIONS = ["Full", "Cut"]

RISK_HIGH = 0.5
RISK_MEDIUM = 0.3


@st.cache_data
def load_and_engineer():
    raw = load_raw(DATA_PATH)
    df = engineer(raw)
    return df


@st.cache_resource
def get_trained_model():
    df = load_and_engineer()
    pipeline, metrics, importance = build_and_train(df)
    return pipeline, metrics, importance


def add_interaction_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Is_Printed_Polyester"] = ((df["Design"] == "Printed") & (df["Fabric"] == "Polyester")).astype(int)
    df["Is_Fleece_Premium"] = ((df["Fabric"] == "Fleece") & (df["ASP_Bucket"].isin(["1300-1599", "1600-1999"]))).astype(int)
    df["Is_Linen_Shirt"] = ((df["Fabric"] == "Linen") & (df["Category"] == "Shirts")).astype(int)
    df["Is_Oversized_Outerwear"] = ((df["Fit"] == "Oversized") & (df["Category"].isin(["Jackets", "Sweaters"]))).astype(int)
    return df


def risk_label(prob: float) -> str:
    if prob >= RISK_HIGH:
        return "🔴 High Risk"
    elif prob >= RISK_MEDIUM:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"


def render_key_findings(df, importance, metrics):
    st.subheader("How the model performs on held-out data")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    c2.metric("Precision", f"{metrics['Precision']:.1%}")
    c3.metric("Recall", f"{metrics['Recall']:.1%}")
    c4.metric("F1 Score", f"{metrics['F1']:.2f}")
    c5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2f}")

    st.subheader("The 4 attribute combinations that drive slow-mover risk")
    st.caption(
        "Each pattern below is grounded in a real apparel-inventory risk cause, "
        "not a random correlation. Odds ratio = how many times more likely a SKU "
        "with this combination is to be a slow-mover, all else equal."
    )

    signal_info = [
        ("Is_Printed_Polyester", "Printed design + Polyester fabric",
         "Cheap synthetic print combo — the look fatigues fast in a fast-fashion cycle."),
        ("Is_Fleece_Premium", "Fleece fabric + ₹1,300+ price",
         "Heavyweight fabric has a short wear-window in a mostly-tropical market; a premium price narrows the buyer pool further."),
        ("Is_Linen_Shirt", "Linen fabric + Shirts category",
         "Formal/niche aesthetic that doesn't fit a Gen-Z fast-fashion casual audience."),
        ("Is_Oversized_Outerwear", "Oversized fit + Jackets/Sweaters",
         "Oversized fits work in tees/hoodies but not outerwear — leads to fit misjudgment and fringe-size overstock."),
    ]

    imp_lookup = importance.set_index("feature")
    for feat, title, reason in signal_info:
        match = [f for f in imp_lookup.index if feat in f]
        if not match:
            continue
        row = imp_lookup.loc[match[0]]
        odds = row["odds_ratio"]
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(title, f"{odds:.1f}x risk")
        with col2:
            st.write(reason)
        st.divider()

    st.subheader("Slow-mover rate by category")
    cat_rate = df.groupby("Category")["Slow_Mover"].mean().sort_values(ascending=False)
    st.bar_chart(cat_rate)


def render_single_scorer(pipeline):
    st.subheader("Score a planned SKU before committing to a buy")
    st.caption("Fill in the planned attributes for a new style. No sales history needed.")

    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Category", CATEGORY_OPTIONS)
        fabric = st.selectbox("Fabric", FABRIC_OPTIONS)
        design = st.selectbox("Design", DESIGN_OPTIONS)
    with col2:
        fit = st.selectbox("Fit", FIT_OPTIONS)
        asp_bucket = st.selectbox("ASP Bucket (₹)", ASP_OPTIONS)
        sleeve = st.selectbox("Sleeve Type", SLEEVE_OPTIONS)
    with col3:
        size_avail = st.selectbox("Planned Size Availability", SIZE_AVAIL_OPTIONS)
        st.write("")
        score_btn = st.button("Score this SKU", type="primary")

    if score_btn:
        input_df = pd.DataFrame([{
            "Category": category, "Fabric": fabric, "Design": design, "Fit": fit,
            "ASP_Bucket": asp_bucket, "Sleeve_Type": sleeve, "Size_Availability": size_avail,
        }])
        input_df = add_interaction_flags(input_df)
        model_input = input_df[ATTRIBUTE_FEATURES + INTERACTION_FEATURES]

        prob = pipeline.predict_proba(model_input)[0, 1]
        st.markdown(f"### {risk_label(prob)} — {prob:.0%} probability of being a slow-mover")
        st.progress(min(prob, 1.0))

        reasons = explain_prediction(pipeline, model_input)
        if reasons:
            st.write("**Top factors pushing this toward slow-mover risk:**")
            for name, contrib in reasons:
                st.write(f"- {friendly_feature_name(name)}")
        else:
            st.write("No strong risk factors identified for this attribute combination.")

        if prob >= RISK_HIGH:
            st.warning("Suggested action: consider a smaller initial buy or a phased/replenishment-based order instead of a full-season commitment.")
        elif prob >= RISK_MEDIUM:
            st.info("Suggested action: monitor closely after launch; not flagged as high-risk but worth watching.")
        else:
            st.success("Suggested action: profile looks safe for a standard full-season buy quantity.")


def render_bulk_scorer(pipeline):
    st.subheader("Score an upcoming collection in bulk")
    st.caption(
        "Upload an Excel file with columns: Category, Fabric, Design, Fit, "
        "ASP_Bucket, Sleeve_Type, Size_Availability (one row per planned SKU)."
    )

    template = pd.DataFrame([{
        "SKU_ID": "PLANNED-001", "Category": "Shirts", "Fabric": "Linen", "Design": "Solid",
        "Fit": "Regular", "ASP_Bucket": "1300-1599", "Sleeve_Type": "Full Sleeve",
        "Size_Availability": "Full",
    }])
    buf = io.BytesIO()
    template.to_excel(buf, index=False)
    st.download_button(
        "Download template", data=buf.getvalue(),
        file_name="planned_skus_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uploaded = st.file_uploader("Upload planned SKU file (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        try:
            user_df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            return

        missing = [c for c in ATTRIBUTE_FEATURES if c not in user_df.columns]
        if missing:
            st.error(f"Missing required column(s): {', '.join(missing)}")
            return

        scored = add_interaction_flags(user_df)
        model_input = scored[ATTRIBUTE_FEATURES + INTERACTION_FEATURES]
        probs = pipeline.predict_proba(model_input)[:, 1]
        scored["Slow_Mover_Probability"] = probs
        scored["Risk_Level"] = scored["Slow_Mover_Probability"].apply(risk_label)
        scored = scored.sort_values("Slow_Mover_Probability", ascending=False)

        st.write(f"### Scored {len(scored)} SKUs")
        c1, c2, c3 = st.columns(3)
        c1.metric("High Risk", int((scored.Slow_Mover_Probability >= RISK_HIGH).sum()))
        c2.metric("Medium Risk", int(((scored.Slow_Mover_Probability >= RISK_MEDIUM) & (scored.Slow_Mover_Probability < RISK_HIGH)).sum()))
        c3.metric("Low Risk", int((scored.Slow_Mover_Probability < RISK_MEDIUM).sum()))

        display_cols = [c for c in user_df.columns if c in scored.columns] + ["Slow_Mover_Probability", "Risk_Level"]
        st.dataframe(
            scored[display_cols].style.format({"Slow_Mover_Probability": "{:.0%}"}),
            use_container_width=True,
        )

        st.subheader("Risk breakdown by category")
        st.bar_chart(scored.groupby("Category")["Slow_Mover_Probability"].mean().sort_values(ascending=False))

        out_buf = io.BytesIO()
        scored.to_excel(out_buf, index=False)
        st.download_button(
            "Download scored results",
            data=out_buf.getvalue(),
            file_name="scored_skus.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def main():
    st.title("👕 Snitch Slow-Mover Prediction Model")
    st.caption("Flag slow-mover risk at the planning stage, before a buy commitment is made.")

    df = load_and_engineer()
    pipeline, metrics, importance = get_trained_model()

    tab1, tab2, tab3 = st.tabs(["🏠 Key Findings", "🔍 Score a Single SKU", "📦 Bulk Portfolio Scoring"])
    with tab1:
        render_key_findings(df, importance, metrics)
    with tab2:
        render_single_scorer(pipeline)
    with tab3:
        render_bulk_scorer(pipeline)

    st.divider()
    st.caption("Trained on a synthetic dataset built to mirror Snitch's real SKU-level sales/stock patterns. Not connected to live sales data.")


if __name__ == "__main__":
    main()
