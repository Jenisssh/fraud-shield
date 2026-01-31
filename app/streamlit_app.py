"""FraudShield Streamlit demo.

Three tabs:

- **Score**   — generate or paste a transaction, hit /predict, see the
              calibrated score + binary decision.
- **Explain** — hit /explain, render SHAP waterfall for the top features.
- **Drift**   — inject synthetic drift and watch PSI light up per feature.

The app talks to the FastAPI service over HTTP (``FRAUDSHIELD_API_URL``,
defaulting to ``http://localhost:8000``). It does *not* import any
model code — that keeps the demo container slim and lets the same
Streamlit app target a remote API.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("FRAUDSHIELD_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 5.0

V_NAMES: list[str] = [f"V{i}" for i in range(1, 29)]


# --------------------------------------------------------------------- helpers
def random_transaction(seed: int | None = None) -> dict[str, float]:
    """Generate a plausible-looking transaction in the ULB schema."""
    rng = np.random.default_rng(seed)
    return {
        "Time": float(rng.uniform(0.0, 172_800.0)),
        **{name: float(rng.normal()) for name in V_NAMES},
        "Amount": float(rng.exponential(scale=80.0)),
    }


def call_predict(transaction: dict[str, float]) -> dict[str, Any] | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=transaction, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        st.error(f"Could not reach API at {API_URL}: {e}")
        return None
    if r.status_code != 200:
        st.error(f"API returned {r.status_code}: {r.text[:200]}")
        return None
    return dict(r.json())


def fetch_health() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return dict(r.json())
    except requests.RequestException:
        pass
    return None


def score_gauge(score: float, threshold: float) -> go.Figure:
    """Plotly indicator showing the score, the chosen threshold, and the band."""
    bar_color = "#dd8452" if score >= threshold else "#4c72b0"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f"},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "P(fraud)"},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".1f"},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, threshold], "color": "#e8eef5"},
                    {"range": [threshold, 1.0], "color": "#fce8da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.85,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(height=280, margin={"t": 40, "b": 10, "l": 30, "r": 30})
    return fig


# --------------------------------------------------------------------- layout
st.set_page_config(page_title="FraudShield", page_icon="🛡", layout="wide")

st.title("FraudShield")
st.caption(
    "Calibrated credit-card fraud detection with cost-aware thresholds, "
    "SHAP explanations, and drift monitoring."
)

with st.sidebar:
    st.header("Service")
    health = fetch_health()
    if health is None:
        st.error("API unreachable")
        st.caption(f"Looked at: {API_URL}")
    else:
        st.success(f"Model {health['model_version']} live")
        st.metric("Active threshold", f"{health['threshold']:.4f}")

    st.divider()
    st.caption(f"API: `{API_URL}`")

tab_score, tab_explain, tab_drift = st.tabs(["Score", "Explain", "Drift"])


# --------------------------------------------------------------------- Tab 1
with tab_score:
    st.subheader("Score a transaction")

    if "transaction" not in st.session_state:
        st.session_state.transaction = random_transaction(seed=0)

    left, right = st.columns([1, 1])

    with left:
        st.write("**Transaction payload**")
        if st.button("Generate random", use_container_width=True):
            st.session_state.transaction = random_transaction()
            st.session_state.pop("last_score", None)
        if st.button("Score this transaction", type="primary", use_container_width=True):
            st.session_state.last_score = call_predict(st.session_state.transaction)
        with st.expander("Show JSON"):
            st.json(st.session_state.transaction, expanded=False)

    with right:
        result = st.session_state.get("last_score")
        if result is None:
            st.info("Click *Score this transaction* to call the API.")
        else:
            st.plotly_chart(
                score_gauge(result["score"], result["threshold"]),
                use_container_width=True,
            )
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                decision = result["decision"]
                emoji = "🔴" if decision == "FRAUD" else "🟢"
                st.metric("Decision", f"{emoji} {decision}")
            with mcol2:
                st.metric("Model version", result["model_version"])


# --------------------------------------------------------------------- Tab 2 (placeholder)
with tab_explain:
    st.subheader("SHAP explanation")
    st.info("Coming in the next commit — wires up POST /explain with a waterfall plot.")


# --------------------------------------------------------------------- Tab 3 (placeholder)
with tab_drift:
    st.subheader("Drift monitoring")
    st.info(
        "Coming in the next commit — PSI heatmap with a 'simulate drift' "
        "slider on top of the training distribution."
    )
