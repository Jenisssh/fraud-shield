# FraudShield

> Calibrated credit card fraud detection with cost-aware thresholds, SHAP explanations,
> and a live drift monitoring dashboard. Trained on the [ULB Credit Card Fraud Detection][ulb]
> dataset (284,807 transactions, 0.172% positive class).

[![CI](https://github.com/Jenisssh/fraud-shield/actions/workflows/ci.yml/badge.svg)](https://github.com/Jenisssh/fraud-shield/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

> **Status:** Day 1 / 20 — project scaffolded. EDA, modeling, and deployment phases are
> tracked in [`docs/roadmap.md`](docs/roadmap.md). This README will fill out as the
> project matures.

---

## Why this exists

Most fraud-detection tutorials stop at "trained XGBoost, got 0.99 ROC-AUC, done." That
number is misleading under 0.17% imbalance, and the resulting model is unusable in
production: its scores aren't calibrated probabilities, the threshold has no business
meaning, and there's no story for what happens when feature distributions shift.

FraudShield is built to demonstrate the *production* concerns:

- **Probability calibration** so that "score = 0.8" actually means "80% likely fraud."
- **Cost-aware threshold tuning** against a configurable `cost(FN) / cost(FP)` matrix.
- **Per-prediction SHAP explanations** served from the inference API.
- **Drift monitoring** with PSI and KS tests on incoming feature distributions.

## Quickstart

```bash
# 1. Editable install + dev tools + pre-commit hooks
make install-dev

# 2. Download the dataset (requires a Kaggle API token — see data/README.md)
make data

# 3. Train (will take ~3 min on a laptop CPU)
make train

# 4. Run the API and the demo app in separate terminals
make serve   # FastAPI on http://localhost:8000/docs
make app     # Streamlit on http://localhost:8501
```

## Project structure

```
fraud-shield/
├── src/fraud_shield/   # installable package — all production code
│   ├── data/           # ingest, schema, splits
│   ├── features/       # sklearn-compatible transformers
│   ├── models/         # baseline, LightGBM, calibration
│   ├── evaluation/     # metrics, threshold, interpretability
│   ├── monitoring/     # drift detection + simulator
│   ├── api/            # FastAPI service
│   └── utils/
├── tests/              # pytest suite (target ≥90% coverage)
├── notebooks/          # exploration only — no production code
├── app/                # Streamlit demo
├── scripts/            # CLI entry points
├── configs/            # YAML configs (data, model, threshold)
├── docs/               # data card, model card, architecture
└── data/               # gitignored; see data/README.md
```

## Tech stack

| Concern | Tool |
|---|---|
| Model | LightGBM + Optuna |
| Calibration | scikit-learn `CalibratedClassifierCV` (isotonic) |
| Explanations | SHAP |
| Validation | pandera + pydantic |
| Drift | PSI, KS-test (custom) |
| Serving | FastAPI + uvicorn |
| Demo | Streamlit |
| Quality | ruff, mypy strict, pytest, pre-commit |
| CI | GitHub Actions |

## Results

_Filled in during Week 2 after the LightGBM model lands. Target: PR-AUC ≥ 0.88 with calibration._

| Model | PR-AUC | ROC-AUC | Recall @ Precision=0.9 | Brier |
|-------|--------|---------|------------------------|-------|
| LogReg (baseline) | TBD | TBD | TBD | TBD |
| LightGBM (raw) | TBD | TBD | TBD | TBD |
| LightGBM + isotonic | TBD | TBD | TBD | TBD |

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for the 4-week plan.

## Acknowledgements

Dataset: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
*Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI, 2015.

## License

MIT — see [LICENSE](LICENSE).

[ulb]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
