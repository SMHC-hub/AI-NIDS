"""AI-NIDS Streamlit dashboard: CSV test-flow replay, SHAP, and optional Groq."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from xai_engine import explain_one, generate_soc_report, load_artifacts

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

st.set_page_config(page_title="AI-NIDS | SOC Dashboard", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background: #0b1220; color: #e5edf8;}
[data-testid="stMetric"] {background: #142136; padding: 16px; border: 1px solid #293b55; border-radius: 12px;}
[data-testid="stSidebar"] {background: #101b2d;}
h1, h2, h3 {letter-spacing: -.02em;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained model...")
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Loading a small test-data sample...")
def get_test_rows(limit: int, feature_names: tuple[str, ...]):
    columns = list(feature_names) + ["Label"]
    scaled = pd.read_csv(DATA_DIR / "test.csv", nrows=limit)
    raw = pd.read_csv(DATA_DIR / "test_raw.csv", nrows=limit)
    if scaled.columns.tolist() != columns or raw.columns.tolist() != columns:
        raise ValueError("Saved test files have unexpected columns or feature order")
    if len(scaled) != len(raw) or not np.array_equal(scaled["Label"].to_numpy(), raw["Label"].to_numpy()):
        raise ValueError("Scaled and raw test rows are not aligned")
    return scaled, raw


st.title("🛡️ AI-NIDS | SOC Command Center")
st.caption("Explainable network intrusion detection • CIC-IDS2017 held-out test flows • CSV replay, not live packet capture")

with st.sidebar:
    st.header("Traffic replay settings")
    row_limit = st.slider("Test flows to load", min_value=20, max_value=1000, value=200, step=20)
    st.caption("Only the first selected number of test rows is read into memory.")
    st.divider()
    st.info("Random Forest detects patterns. SHAP explains one prediction. Groq writes an optional defensive report.")

try:
    model, scaler, features = get_artifacts()
    scaled, raw = get_test_rows(row_limit, tuple(features))
    if scaled.empty:
        st.error("No test rows found.")
        st.stop()
    X = scaled[features].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Test features contain non-finite values")
    predictions = model.predict(X).astype(int)
    probabilities = model.predict_proba(X)
    malicious_index = list(model.classes_).index(1)
    malicious_probs = probabilities[:, malicious_index]
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Cannot load dashboard data: {exc}")
    st.stop()

malicious_count = int((predictions == 1).sum())
benign_count = int((predictions == 0).sum())
col1, col2, col3, col4 = st.columns(4)
col1.metric("Flows analyzed", f"{len(predictions):,}")
col2.metric("Predicted malicious", f"{malicious_count:,}")
col3.metric("Predicted benign", f"{benign_count:,}")
col4.metric("Malicious prediction rate", f"{malicious_count / len(predictions):.1%}")

st.subheader("Traffic classification")
chart_col, table_col = st.columns([1, 2])
with chart_col:
    st.bar_chart(pd.DataFrame({"Flows": [benign_count, malicious_count]}, index=["Benign", "Malicious"]))
with table_col:
    overview = pd.DataFrame({
        "Row": np.arange(len(predictions)),
        "Actual label": np.where(scaled["Label"].to_numpy() == 1, "Malicious", "Benign"),
        "Prediction": np.where(predictions == 1, "Malicious", "Benign"),
        "P(malicious)": np.round(malicious_probs, 4),
    })
    st.dataframe(overview, use_container_width=True, height=280, hide_index=True)

st.divider()
st.subheader("Inspect an individual flow")
filter_choice = st.radio("Show rows", ["All", "Predicted malicious", "Predicted benign"], horizontal=True)
indices = np.arange(len(predictions))
if filter_choice == "Predicted malicious":
    indices = indices[predictions == 1]
elif filter_choice == "Predicted benign":
    indices = indices[predictions == 0]
if len(indices) == 0:
    st.warning("No flows match this filter. Choose another filter.")
    st.stop()

selected = st.selectbox("Select test row", options=indices.tolist(), format_func=lambda i: f"Row {i} · {'MALICIOUS' if predictions[i] == 1 else 'BENIGN'} · P(malicious)={malicious_probs[i]:.2%}")
predicted = int(predictions[selected])
probability = float(probabilities[selected, list(model.classes_).index(predicted)])
st.markdown(f"**Prediction:** {'🔴 MALICIOUS' if predicted == 1 else '🟢 BENIGN'}  |  **Predicted-class probability:** {probability:.2%}  |  **Dataset label:** {int(scaled.iloc[selected]['Label'])}")
st.caption("A model probability is not proof of an attack. Test-set results may not reflect real-world performance.")

with st.expander("View raw network-flow features"):
    st.dataframe(pd.DataFrame({"Feature": features, "Raw value": raw.iloc[selected][features].to_numpy()}), hide_index=True, use_container_width=True)

if st.button("🔍 Explain this prediction with SHAP", type="primary"):
    with st.spinner("Computing SHAP values for one flow..."):
        try:
            result_pred, result_prob, details = explain_one(
                model, features, X[selected], raw.iloc[selected], top_k=5
            )
            st.session_state["explanation"] = {
                "row": int(selected), "predicted": int(result_pred),
                "probability": float(result_prob), "details": details,
            }
            st.session_state.pop("soc_report", None)
        except Exception as exc:
            st.error(f"SHAP explanation failed ({type(exc).__name__}): {exc}")

explanation = st.session_state.get("explanation")
if explanation and explanation["row"] == int(selected):
    st.subheader("Local SHAP explanation")
    details_df = pd.DataFrame(explanation["details"])
    st.dataframe(details_df.rename(columns={"feature": "Feature", "raw_value": "Raw value", "shap_value": "SHAP value", "direction": "Direction"}), hide_index=True, use_container_width=True)
    st.bar_chart(details_df.set_index("feature")["shap_value"])
    st.caption("SHAP values show contribution to the predicted class for this flow, not causality.")

    if st.button("🤖 Generate Groq SOC Analyst report"):
        with st.spinner("Requesting defensive analysis from Groq..."):
            st.session_state["soc_report"] = generate_soc_report(
                explanation["predicted"], explanation["probability"], explanation["details"]
            )
    if st.session_state.get("soc_report"):
        st.subheader("Groq SOC Analyst report")
        st.markdown(st.session_state["soc_report"])
        st.caption("AI-generated investigative guidance. Validate against logs and additional evidence before taking action.")
else:
    st.info("Click 'Explain this prediction with SHAP' to see feature contributions and enable the optional Groq report.")

st.divider()
st.caption("AI-NIDS portfolio prototype | CIC-IDS2017 | Random Forest + SHAP + Groq")
