# FraudShield

> Calibrated credit-card fraud detection with cost-aware thresholds,
> SHAP explanations, and a live drift-monitoring dashboard. Trained on
> the [ULB Credit Card Fraud Detection][ulb] dataset (284,807
> transactions, 0.172% positive class).

[![CI](https://github.com/Jenisssh/fraud-shield/actions/workflows/ci.yml/badge.svg)](https://github.com/Jenisssh/fraud-shield/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![Type-checked: mypy](https://img.shields.io/badge/mypy-strict-2a6db4.svg)](https://mypy.readthedocs.io/)

---

## What this is

Most fraud-detection projects stop at *"trained XGBoost, got 0.99
ROC-AUC, done."* That number is misleading at 0.17% imbalance and the
resulting model is unusable in production: its scores aren't
calibrated probabilities, its threshold has no business meaning, and
nothing watches for distribution shift after deployment.

FraudShield is built around the *production* concerns:

- **Probability calibration** so that `score = 0.8` actually means
  "80% likely fraud" — verified with reliability diagrams and Brier
  score.
- **Cost-aware threshold tuning** against a configurable
  `C_FN * FN + C_FP * FP` matrix, not F1 or accuracy.
- **Per-prediction SHAP explanations** served by the FastAPI
  `/explain` endpoint and rendered as a waterfall plot in the
  Streamlit demo.
- **Drift monitoring** with PSI + KS-test per feature, with a
  drift-injection simulator powering the dashboard slider.

## Quickstart

```bash
# 1. Editable install + dev tools + pre-commit hooks
make install-dev

# 2. Download the ULB dataset (needs a Kaggle API token)
make data

# 3. Train the LightGBM + isotonic pipeline (~3 min on a laptop CPU)
make train

# 4. Run the API and the Streamlit demo in separate terminals
make serve    # FastAPI on http://localhost:8000/docs
make app      # Streamlit on http://localhost:8501

# Or do everything in containers:
docker compose up
```

## Architecture

```
data ─► schema (pandera) ─► splits ─► features ─► LightGBM ─► calibration
                                                                  │
                              ┌───────────────────────────────────┤
                              ▼                                   ▼
                       cost-aware threshold              SHAP TreeExplainer
                              │                                   │
                              └──────────► FastAPI ◄──────────────┘
                                              │
                            ┌─────────────────┼──────────────────┐
                            ▼                                    ▼
                       Streamlit demo                       monitoring (PSI + KS)
```

Full diagram + request paths in [`docs/architecture.md`](docs/architecture.md).
Model details and limitations in [`docs/model_card.md`](docs/model_card.md).
Dataset provenance in [`docs/data_card.md`](docs/data_card.md).

## Results

Measured on the held-out 15% test partition (stratified, seed 42).
Populated by the final cell of
[`notebooks/04_calibration_and_thresholds.ipynb`](notebooks/04_calibration_and_thresholds.ipynb)
after a real training run.

| Model | PR-AUC | ROC-AUC | Recall @ Precision=0.9 | Brier |
|-------|--------|---------|-----------------------|-------|
| Logistic baseline (`class_weight='balanced'`) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| LightGBM (raw) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| LightGBM + isotonic | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Headline metric:** PR-AUC. ROC-AUC is reported for completeness
but under-discriminates at this imbalance.

## Project structure

```
fraud-shield/
├── src/fraud_shield/         # installable package — production code
│   ├── data/                 # ingest, schema, splits
│   ├── features/             # sklearn-compatible TimeAmountFeatures
│   ├── models/               # baseline, LightGBM, tuning, calibration
│   ├── evaluation/           # metrics, threshold, SHAP interpretability
│   ├── monitoring/           # drift detection + simulator
│   ├── api/                  # FastAPI service
│   └── utils/                # structlog setup
├── tests/                    # 208 tests, mypy-strict
├── notebooks/                # 01 EDA · 02 correlations · 03 imbalance · 04 calibration+threshold
├── app/streamlit_app.py      # Score / Explain / Drift tabs
├── scripts/                  # CLI entry points
├── docs/                     # model card, data card, architecture
└── data/                     # gitignored; see data/README.md
```

## Tech stack

| Concern | Tool |
|---|---|
| Model | LightGBM + Optuna |
| Calibration | isotonic regression (custom wrapper) |
| Explanations | SHAP TreeExplainer |
| Validation | pandera, pydantic |
| Drift | PSI + KS-test (custom) |
| Serving | FastAPI + uvicorn |
| Demo | Streamlit + Plotly |
| Quality | ruff, mypy strict, pytest |
| Container | Docker (multi-stage, non-root) |
| CI | GitHub Actions |

## Interview talking points

1. **Why PR-AUC over ROC-AUC?** ROC-AUC's FPR axis is saturated at
   0.17% positive rate; PR-AUC focuses on the minority class, which
   is what fraud teams act on.
2. **How do I handle imbalance?** Loss-side weighting (`class_weight`
   for LR, `scale_pos_weight` for LightGBM). Notebook 03 compares this
   to SMOTE and undersampling. SMOTE matches the recall lift but
   degrades Brier score — that's the calibration tradeoff most papers
   skip.
3. **Why calibrate?** Raw gradient-boosted scores aren't
   probabilities, they're rankings. Isotonic regression on the val
   set turns them into something a business rule like "block at
   P > 0.8" can actually consume.
4. **How do I pick a threshold?** Not by F1 — by expected cost. See
   `src/fraud_shield/evaluation/threshold.py`. The cost matrix is
   configurable so the deployment owner moves the lever, not the
   model author.
5. **How do I monitor it in production?** Three layers:
   (a) feature drift via PSI per feature, alert at PSI > 0.2;
   (b) prediction drift via KS test on score distributions;
   (c) label drift once chargeback feedback arrives.
   Layers (a) and (b) are implemented in `monitoring/drift.py`; the
   Streamlit Drift tab demoes them live.
6. **What would I do with another month?** Online learning loop with
   chargeback feedback, IEEE-CIS dataset for richer features,
   Shapley-value sampling to make `/explain` faster, and a real A/B
   test harness against the current rules engine.

## Acknowledgements

Dataset: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and
Gianluca Bontempi. *Calibrating Probability with Undersampling for
Unbalanced Classification.* IEEE SSCI, 2015.

## License

MIT — see [LICENSE](LICENSE).

[ulb]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
