# fraud-shield

Credit card fraud detection on the ULB dataset. LightGBM with isotonic
calibration, cost-aware threshold tuning, and a small FastAPI + Streamlit
demo on top.

[![CI](https://github.com/Jenisssh/fraud-shield/actions/workflows/ci.yml/badge.svg)](https://github.com/Jenisssh/fraud-shield/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## What's in here

The problem is the 0.17% positive rate, not the model. So the project spends
most of its effort on:

- pandera-validated data loading, stratified + time-aware splits
- a logistic regression baseline so the more complex model has something to beat
- LightGBM with Optuna hyperparameter search
- isotonic probability calibration (compared against Platt) on a held-out val set
- threshold optimization against a configurable C_FN / C_FP cost matrix
- SHAP explanations for individual predictions
- PSI and KS-test based drift detection, with a small simulator for the demo
- a FastAPI service exposing `/predict` and `/explain`
- a Streamlit dashboard with three tabs (Score, Explain, Drift)

## Running it

You need Python 3.12 and a Kaggle API token for the dataset.

```
make install-dev    # editable install + dev tools
make data           # downloads creditcard.csv (~144 MB)
make train          # trains and persists the calibrated model
make serve          # FastAPI on :8000
make app            # Streamlit on :8501
```

Or `docker compose up` to bring up both services.

## Layout

```
src/fraud_shield/
  data/         ingest, pandera schema, train/val/test splits
  features/     sklearn transformer for Time / Amount
  models/       baseline, LightGBM, Optuna tuning, calibration
  evaluation/   metrics, threshold optimization, SHAP wrapper
  monitoring/   drift detection + simulator
  api/          FastAPI app

notebooks/   01 EDA, 02 correlations, 03 imbalance, 04 calibration + threshold
app/         Streamlit demo
tests/       pytest suite, mypy strict, ruff clean
docs/        model card, data card, architecture
```

## Results

Filled in by the last cell of `notebooks/04_calibration_and_thresholds.ipynb`
after you run `make train`. The expected pattern on this dataset is roughly
PR-AUC around 0.74 for the LR baseline, around 0.86 for raw LightGBM, and
0.87–0.88 after isotonic calibration with a clear Brier improvement.

## A few notes on choices

PR-AUC, not ROC-AUC. ROC-AUC's FPR axis is dominated by the negative class
at this imbalance and barely separates obviously different models.

Loss-side weighting (`class_weight='balanced'` for LR,
`scale_pos_weight=neg/pos` for LightGBM) instead of SMOTE. Notebook 03 walks
through why — SMOTE matches the recall lift but degrades calibration, which
defeats the point of having calibrated probabilities downstream.

Cost-aware threshold. F1 doesn't say anything about whether a missed fraud
costs more than a false alarm. The optimizer in `evaluation/threshold.py`
takes a `CostMatrix(fn, fp)` and finds the threshold that minimizes
expected cost.

SHAP values on the margin (log-odds) scale, with the additivity property
checked end-to-end in tests. The API surfaces both the margin and the
calibrated score so downstream callers can decide which to consume.

## Dataset

ULB Credit Card Fraud Detection (Dal Pozzolo et al., 2015). 284,807
transactions, 492 fraud cases, two days in September 2013. Kaggle:
`mlg-ulb/creditcardfraud`. ODbL license.

## License

MIT.
