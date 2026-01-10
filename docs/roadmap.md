# Roadmap

A 4-week, ~10 hrs/week plan. Each row is one focused work session that ends
in 2–5 small commits.

## Week 1 — Foundations & EDA

| Day | Task | Output |
|-----|------|--------|
| 1   | Scaffold the project — pyproject, lint, types, tests, CI, license, README skeleton | First commits, CI green |
| 2   | Kaggle download script + data card + EDA notebook part 1 (class balance, distributions, time patterns) | `notebooks/01_eda.ipynb` |
| 3   | EDA part 2 (correlations, fraud-vs-normal features), pandera schema | `src/fraud_shield/data/schema.py` |
| 4   | Stratified + time-aware train/val/test splits with tests | `src/fraud_shield/data/splits.py` |
| 5   | Time and amount feature transformers (sklearn API) with tests | `src/fraud_shield/features/` |

## Week 2 — Modeling & Calibration

| Day | Task | Output |
|-----|------|--------|
| 6   | Logistic regression baseline + metrics module (PR-AUC, recall@p=0.9) | `models/baseline.py`, `evaluation/metrics.py` |
| 7   | LightGBM trainer with early stopping + Optuna search (≤50 trials, 20 min) | `models/lightgbm_model.py` |
| 8   | Imbalance handling notebook: `scale_pos_weight` vs SMOTE vs undersampling | `notebooks/03_baseline_vs_boosted.ipynb` |
| 9   | Probability calibration (isotonic vs Platt) + reliability diagram | `models/calibration.py` |
| 10  | Cost-aware threshold optimization with configurable cost matrix | `evaluation/threshold.py` |

## Week 3 — Interpretation, Drift, Service

| Day | Task | Output |
|-----|------|--------|
| 11  | SHAP global + per-prediction explanations | `evaluation/interpretability.py` |
| 12  | Drift module: PSI + KS test per feature, with a drift simulator | `monitoring/drift.py`, `monitoring/simulate.py` |
| 13  | FastAPI service: `/health`, `/predict`, `/explain` endpoints | `api/main.py` |
| 14  | API integration tests (TestClient), request validation, error responses | `tests/test_api.py` |
| 15  | Dockerfile (multi-stage), docker-compose with smoke test | `Dockerfile`, `docker-compose.yml` |

## Week 4 — Demo, Docs, Polish

| Day | Task | Output |
|-----|------|--------|
| 16  | Streamlit tab 1 — score a transaction | `app/streamlit_app.py` |
| 17  | Streamlit tab 2 — SHAP waterfall + tab 3 — drift dashboard | App complete |
| 18  | Model card + architecture diagram | `docs/model_card.md`, `docs/architecture.md` |
| 19  | README polish — 60-sec pitch, results table, demo GIF | `README.md` final |
| 20  | Deploy: Streamlit Cloud + Fly.io API, record demo video | Live URLs in README |

## Stretch goals (post-week-4, optional)

- IEEE-CIS Fraud Detection — richer features, more complex preprocessing
- Online learning loop with simulated chargeback feedback
- Prometheus metrics + Grafana dashboard for the API
- A/B test harness comparing calibrated vs raw thresholds

## Definition of Done

- All CI checks pass on `main`
- PR-AUC ≥ 0.85 on the held-out test set, reproducible via `make train`
- pytest coverage ≥ 90% on `src/fraud_shield`
- Streamlit demo loads and all three tabs work end-to-end
- README quickstart works on a fresh clone within 10 minutes
