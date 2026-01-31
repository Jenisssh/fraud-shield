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


def call_explain(transaction: dict[str, float]) -> dict[str, Any] | None:
    try:
        r = requests.post(f"{API_URL}/explain", json=transaction, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        st.error(f"Could not reach API at {API_URL}: {e}")
        return None
    if r.status_code != 200:
        st.error(f"API returned {r.status_code}: {r.text[:200]}")
        return None
    return dict(r.json())


def shap_waterfall(
    expected_value: float,
    contributions: list[dict[str, Any]],
) -> go.Figure:
    """Plotly waterfall: expected_value → margin, with one bar per top feature."""
    measures = ["absolute"] + ["relative"] * len(contributions) + ["total"]
    x_labels = (
        ["E[score]"] + [f"{c['feature']} = {c['value']:.3f}" for c in contributions] + ["margin"]
    )
    y_values: list[float | None] = [expected_value]
    y_values.extend(float(c["shap_value"]) for c in contributions)
    y_values.append(None)  # total is computed by Plotly

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=x_labels,
            y=y_values,
            connector={"line": {"color": "#888", "width": 1}},
            increasing={"marker": {"color": "#dd8452"}},
            decreasing={"marker": {"color": "#4c72b0"}},
            totals={"marker": {"color": "#222"}},
        )
    )
    fig.update_layout(
        height=420,
        margin={"t": 30, "b": 30, "l": 40, "r": 30},
        xaxis_tickangle=-30,
        yaxis_title="log-odds",
        showlegend=False,
    )
    return fig


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


# --------------------------------------------------------------------- Tab 2
with tab_explain:
    st.subheader("SHAP explanation")
    st.caption(
        "Each bar is one feature's contribution to the model's margin (log-odds). "
        "Orange pushes the prediction toward FRAUD, blue toward OK. "
        "Sum of bars + the base value = margin, which becomes the calibrated "
        "score after the sigmoid."
    )

    if st.button("Explain current transaction", type="primary"):
        st.session_state.last_explain = call_explain(st.session_state.transaction)

    explain = st.session_state.get("last_explain")
    if explain is None:
        st.info(
            "Click *Explain current transaction* to fetch SHAP values for the "
            "transaction shown in the Score tab."
        )
    else:
        top_k = st.slider("Show top-k features", min_value=3, max_value=10, value=8)
        top = explain["top_contributions"][:top_k]

        st.plotly_chart(
            shap_waterfall(explain["expected_value"], top),
            use_container_width=True,
        )

        st.caption(
            f"Score: **{explain['score']:.3f}**  •  "
            f"Decision: **{explain['decision']}**  •  "
            f"Threshold: {explain['threshold']:.4f}  •  "
            f"Margin: {explain['raw_score']:.3f}"
        )

        with st.expander("Full contributions table"):
            st.dataframe(
                explain["top_contributions"],
                column_config={
                    "feature": "Feature",
                    "value": st.column_config.NumberColumn("Value", format="%.4f"),
                    "shap_value": st.column_config.NumberColumn("SHAP", format="%.4f"),
                },
                hide_index=True,
                use_container_width=True,
            )


# --------------------------------------------------------------------- Tab 3 (placeholder)
with tab_drift:
    st.subheader("Drift monitoring")
    st.info(
        "Coming in the next commit — PSI heatmap with a 'simulate drift' "
        "slider on top of the training distribution."
    )
