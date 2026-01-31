# Architecture

A one-page mental model of the system.

## Component layout

```
                                ┌──────────────────┐
                                │  data/raw/       │
                                │  creditcard.csv  │
                                │  (gitignored)    │
                                └────────┬─────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  fraud_shield.data  │
                              │  ─ ingest           │
                              │  ─ schema (pandera) │
                              │  ─ splits           │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  features.          │
                              │  TimeAmountFeatures │
                              └──────────┬──────────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
   ┌────────▼────────┐         ┌─────────▼─────────┐         ┌────────▼─────────┐
   │ models.baseline │         │ models.lightgbm   │         │ models.tuning    │
   │ LogReg (LR)     │         │ LightGBMClassifier│  ◄────  │ Optuna TPE       │
   │ class_weight=   │         │ scale_pos_weight= │         │ search           │
   │   balanced      │         │   auto            │         │                  │
   └─────────────────┘         └─────────┬─────────┘         └──────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ models.calibration  │
                              │ CalibratedFraud...  │
                              │ (isotonic | Platt)  │
                              └──────────┬──────────┘
                                         │
                  ┌──────────────────────┼──────────────────────┐
                  │                      │                      │
       ┌──────────▼──────────┐ ┌─────────▼────────┐  ┌─────────▼──────────┐
       │ evaluation.metrics  │ │ evaluation.      │  │ evaluation.        │
       │ PR-AUC, recall@p,   │ │   threshold       │  │   interpretability │
       │ Brier               │ │ CostMatrix +      │  │ SHAP TreeExplainer │
       └─────────────────────┘ │ optimize_thresh.  │  └────────────────────┘
                               └──────────┬────────┘
                                          │
                                          │ artifacts written to models/
                                          ▼
                         ┌─────────────────────────────────┐
                         │  models/                        │
                         │  ─ calibrated.joblib            │
                         │     {model, explainer, version} │
                         │  ─ threshold.txt                │
                         └────────────────┬────────────────┘
                                          │ loaded once via lifespan
                                          ▼
                       ┌──────────────────────────────────┐
                       │  api/main.py  (FastAPI)          │
                       │  ─ /health                       │
                       │  ─ /predict                      │
                       │  ─ /explain                      │
                       └─────┬──────────────────┬─────────┘
                             │                  │
              ┌──────────────▼─────┐  ┌─────────▼──────────┐
              │  app/streamlit_app │  │  monitoring.drift  │
              │  Score / Explain / │  │  PSI + KS,         │
              │  Drift tabs        │  │  inject_drift      │
              └────────────────────┘  └────────────────────┘
```

## Request paths

**`POST /predict`**

```
   client
     │
     │  Transaction JSON (Time, V1..V28, Amount)
     ▼
  Pydantic schema validation
     │   (422 on missing/extra/negative)
     ▼
  pd.DataFrame ─►  artifacts.model.predict_proba
                          │
                          ▼
                 score ─►  decision (score ≥ threshold)
                          │
                          ▼
                  PredictResponse JSON
```

**`POST /explain`**

```
   client
     │
     │  Transaction JSON
     ▼
  Pydantic validation
     │
     ▼
  pd.DataFrame ─►  artifacts.model.predict_proba      (score, decision)
              \─►  artifacts.explainer.explain_one    (SHAP values)
                          │
                          ▼
                  top-10 contributions sorted by |SHAP|
                          │
                          ▼
                  ExplainResponse JSON
```

## Build & deploy

- **Local dev:** `make install-dev`, `make serve` + `make app` in two terminals.
- **Container:** `docker compose up` builds both images and brings up
  the API + Streamlit stack on `fraud-shield-net`. The API mounts
  `./models` read-only.
- **Production targets (future work):** Fly.io for the API (single
  region, shared CPU), Streamlit Community Cloud for the demo. CI on
  release tag pushes the API image to a registry.

## Reproducibility

- Fixed `random_seed=42` everywhere (`fraud_shield.config.settings`).
- LightGBM `deterministic=True, force_col_wise=True` — bit-stable
  reruns across machines.
- Dependencies pinned in `requirements.txt` (runtime),
  `requirements-dev.txt` (dev), `requirements-notebook.txt` (notebooks).
- Snapshot date documented in each lock file.

## Testing surface

```
tests/
├── test_data.py            (covered by test_schema, test_splits)
├── test_schema.py           pandera validation
├── test_splits.py           stratified + time-aware
├── test_transformers.py     TimeAmountFeatures
├── test_baseline.py         LR pipeline + class_weight effect
├── test_lightgbm_model.py   wrapper + early stopping + auto SPW
├── test_tuning.py           Optuna study + reproducibility
├── test_calibration.py      isotonic vs Platt vs raw Brier
├── test_threshold.py        CostMatrix + optimum directional checks
├── test_interpretability.py SHAP additivity, top-k
├── test_drift.py            PSI + KS
├── test_simulate.py         drift injection + end-to-end detector loop
├── test_api.py              FastAPI integration (TestClient + stubs)
├── test_api_smoke.py        structural smoke
├── test_metrics.py          PR-AUC, recall@p, Brier corner cases
└── test_download_data.py    Kaggle download + checksum
```

200+ tests, mypy strict, ruff clean, ~90%+ coverage on `src/fraud_shield`.
