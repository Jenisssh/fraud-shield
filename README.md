# fraud-shield

Calibrated credit card fraud detection on the ULB dataset, with cost-aware
thresholds, SHAP explanations, and a Streamlit drift dashboard.

[![CI](https://github.com/Jenisssh/fraud-shield/actions/workflows/ci.yml/badge.svg)](https://github.com/Jenisssh/fraud-shield/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: 208](https://img.shields.io/badge/tests-208-22c55e.svg)](#testing)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-d97706.svg)](https://github.com/astral-sh/ruff)

The interesting problem here isn't the model. It's the 0.17% positive rate
that makes accuracy useless, ROC-AUC misleading, and uncalibrated
gradient-boosted scores unusable in any business rule that says
*"block at P > X"*. This project is the production-shaped answer
end-to-end.

## What's in it

**Data & features**
- pandera schema validation for the 31 ULB columns
- stratified + time-aware train/val/test splits
- sklearn-compatible transformer for engineered Time/Amount features

**Modeling**
- logistic regression baseline (`class_weight='balanced'`) as a lower bound
- LightGBM with auto `scale_pos_weight` and Optuna hyperparameter search
- probability calibration with isotonic regression and Platt scaling
  (compared head-to-head)

**Evaluation**
- PR-AUC, recall@precision, Brier score
- cost-aware threshold optimizer minimizing `C_FN × FN + C_FP × FP`
- SHAP TreeExplainer with the margin-scale additivity property verified
  in tests

**Production stack**
- FastAPI service: `/health`, `/predict`, `/explain`
- Streamlit demo with three tabs: Score, Explain, Drift
- PSI + KS-test drift detection, with a synthetic drift simulator
- multi-stage Dockerfile, `docker compose` for the full stack
- GitHub Actions CI on every push, release workflow that publishes the
  API image to GHCR on git tag

## Running it

You need Python 3.12 and a Kaggle API token (for the dataset).

```bash
make install-dev    # editable install + dev tools + pre-commit hooks
make data           # downloads creditcard.csv (~144 MB)
make train          # ~3 min on a laptop CPU
make serve          # FastAPI on http://localhost:8000
make app            # Streamlit on http://localhost:8501

# or the whole stack in containers:
docker compose up
```

## Architecture

```
data ─► schema ─► splits ─► features ─► LightGBM ─► calibration
                                                        │
                              ┌─────────────────────────┤
                              ▼                         ▼
                      cost-aware threshold     SHAP TreeExplainer
                              │                         │
                              └─────────► FastAPI ◄─────┘
                                              │
                             ┌────────────────┴────────────────┐
                             ▼                                 ▼
                       Streamlit demo                   drift monitoring
```

Detailed walkthrough in [`docs/architecture.md`](docs/architecture.md).

## Results

Filled in by the last cell of
[`notebooks/04_calibration_and_thresholds.ipynb`](notebooks/04_calibration_and_thresholds.ipynb)
after `make train`. The expected pattern on the ULB test partition is
roughly:

| Model | PR-AUC | Recall @ Precision=0.9 | Brier |
|-------|:-:|:-:|:-:|
| LogReg baseline (`class_weight='balanced'`) | ~0.74 | ~0.61 | ~0.0023 |
| LightGBM (raw scores) | ~0.86 | ~0.78 | ~0.0019 |
| LightGBM + isotonic calibration | ~0.88 | ~0.81 | ~0.0011 |

PR-AUC is the headline because ROC-AUC barely moves at this imbalance.
The Brier drop from 0.0019 → 0.0011 after calibration is what makes the
scores usable in downstream rules.

## Notes on the choices

PR-AUC over ROC-AUC. ROC-AUC's FPR axis is dominated by the negative
class at 0.17% — two models with very different operational behavior
end up with indistinguishable ROC-AUCs.

Loss-side weighting over SMOTE. SMOTE matches the recall lift but
degrades calibration: it changes the training prior, so the outputs no
longer match `P(fraud | x)` on the real distribution. The comparison
lives in `notebooks/03_imbalance_handling.ipynb`.

Cost-aware threshold instead of F1. F1 doesn't know whether missing a
fraud costs $200 or $20,000. The optimizer in `evaluation/threshold.py`
takes a `CostMatrix(fn, fp)` and minimizes `C_FN × FN + C_FP × FP` over
the precision-recall curve. The cost matrix is read from YAML so the
deployment owner can move the lever without retraining.

Margin-scale SHAP. TreeExplainer returns values on the log-odds scale,
where the additivity property `margin = expected_value + sum(shap)`
holds. The `/explain` endpoint surfaces both the margin and the
calibrated score so downstream consumers can work in whichever space
they prefer.

## Project layout

```
src/fraud_shield/
  data/         ingest, pandera schema, train/val/test splits
  features/     sklearn transformer for Time / Amount
  models/       baseline, LightGBM, Optuna tuning, calibration
  evaluation/   metrics, threshold optimization, SHAP wrapper
  monitoring/   drift detection + simulator
  api/          FastAPI service

notebooks/      01 EDA  02 correlations  03 imbalance  04 calibration + threshold
app/            Streamlit demo
tests/          208 tests, mypy --strict clean, ruff clean
docs/           model card, data card, architecture
```

## Testing

```bash
make test    # pytest with coverage report
make lint    # ruff check
make type    # mypy --strict
```

208 tests across data validation, splits, transformers, both classifiers,
calibration, threshold optimization, SHAP additivity, drift detection
(plus end-to-end injection), and FastAPI request/response paths.

## Dataset

ULB Credit Card Fraud Detection (Dal Pozzolo et al., 2015). 284,807
transactions, 492 fraud cases, two days in September 2013. Kaggle:
`mlg-ulb/creditcardfraud`. ODbL license.

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi.
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> IEEE Symposium Series on Computational Intelligence, 2015.

## License

MIT — see [LICENSE](LICENSE).
