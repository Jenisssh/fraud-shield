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
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from fraud_shield.monitoring.drift import detect_drift
from fraud_shield.monitoring.simulate import inject_drift

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


# --------------------------------------------------------------------- Tab 3
@st.cache_data
def baseline_reference(n: int = 2_000) -> pd.DataFrame:
    """Synthetic reference distribution standing in for the training set.

    Cached so the slider doesn't redraw it every interaction.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({name: rng.normal(size=n) for name in V_NAMES})
    df["Amount"] = rng.exponential(scale=80.0, size=n)
    return df


with tab_drift:
    st.subheader("Drift monitoring")
    st.caption(
        "Compare a synthetic 'production window' against a stable reference "
        "distribution. Move the slider to inject covariate drift into a chosen "
        "feature and watch PSI cross the alert threshold in real time."
    )

    reference = baseline_reference()

    control_a, control_b = st.columns([1, 2])
    with control_a:
        drift_feature = st.selectbox(
            "Feature to drift",
            options=list(reference.columns),
            index=list(reference.columns).index("V14"),
        )
    with control_b:
        magnitude = st.slider(
            "Drift magnitude (standard deviations)",
            min_value=0.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
        )

    current = inject_drift(reference, [drift_feature], kind="mean_shift", magnitude=magnitude)
    report = detect_drift(reference, current, psi_threshold=0.2)

    n_drifted = int(report["drifted"].sum())
    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.metric("Features drifted", n_drifted, delta=f"of {len(report)}")
    with metric_b:
        st.metric("Max PSI", f"{report['psi'].max():.3f}")
    with metric_c:
        focus = report[report["feature"] == drift_feature].iloc[0]
        st.metric(
            f"PSI({drift_feature})",
            f"{focus['psi']:.3f}",
            delta=f"KS p={focus['ks_pvalue']:.3g}",
            delta_color="off",
        )

    bar_colors = ["#dd8452" if d else "#4c72b0" for d in report["drifted"]]
    fig = go.Figure(
        go.Bar(
            x=report["feature"],
            y=report["psi"],
            marker_color=bar_colors,
            hovertemplate=("feature=%{x}<br>" "PSI=%{y:.4f}<br>" "<extra></extra>"),
        )
    )
    fig.add_hline(
        y=0.2,
        line_dash="dash",
        line_color="#dd8452",
        annotation_text="alert threshold (PSI = 0.2)",
        annotation_position="top right",
    )
    fig.update_layout(
        title="PSI per feature (orange = drifted)",
        yaxis_title="PSI",
        xaxis_tickangle=-45,
        height=420,
        margin={"t": 50, "b": 80, "l": 40, "r": 30},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full drift report"):
        st.dataframe(
            report,
            column_config={
                "feature": "Feature",
                "psi": st.column_config.NumberColumn("PSI", format="%.4f"),
                "ks_statistic": st.column_config.NumberColumn("KS stat", format="%.4f"),
                "ks_pvalue": st.column_config.NumberColumn("KS p-value", format="%.3g"),
                "drifted": st.column_config.CheckboxColumn("Alert"),
            },
            hide_index=True,
            use_container_width=True,
        )
