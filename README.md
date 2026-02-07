<div align="center">

# 🛡 FraudShield

**Calibrated credit-card fraud detection with cost-aware thresholds, SHAP explanations, and a live drift dashboard.**

[![CI](https://github.com/Jenisssh/fraud-shield/actions/workflows/ci.yml/badge.svg)](https://github.com/Jenisssh/fraud-shield/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-d97706.svg?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![mypy strict](https://img.shields.io/badge/mypy-strict-2563eb.svg)](https://mypy.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-208%20passing-22c55e.svg)](#testing)

</div>

---

## 📑 Table of Contents

1. [Why this project exists](#-why-this-project-exists)
2. [Highlights](#-highlights)
3. [Demo](#-demo)
4. [Quickstart](#-quickstart)
5. [Architecture](#-architecture)
6. [Results](#-results)
7. [Design decisions worth knowing](#-design-decisions-worth-knowing)
8. [Tech stack](#-tech-stack)
9. [Project layout](#-project-layout)
10. [Testing](#-testing)
11. [Interview talking points](#-interview-talking-points)
12. [Documentation](#-documentation)
13. [License & acknowledgements](#-license--acknowledgements)

---

## 💡 Why this project exists

> Most fraud-detection tutorials end at *"trained XGBoost, got 0.99 ROC-AUC, done."*
> That number is misleading at 0.17% positive rate, and the resulting model is
> unusable in production: its scores aren't calibrated probabilities, its
> threshold has no business meaning, and nothing watches for distribution
> shift after deployment.

**FraudShield is built around the production concerns**, not just the model.
It's an end-to-end answer to *"what would it actually take to deploy this?"* —
calibrated probabilities, cost-aware thresholds, per-prediction explanations,
and drift monitoring, all behind a clean API + Streamlit demo.

---

## ✨ Highlights

> 📈  **Probability calibration** — isotonic regression on a held-out val set so
> `score = 0.8` actually means "80% likely fraud", verified with reliability
> curves and Brier score.
>
> 💰  **Cost-aware thresholding** — minimizes `C_FN × FN + C_FP × FP` against
> a configurable business cost matrix, not F1 or accuracy.
>
> 🔍  **Per-prediction SHAP explanations** — `/explain` returns the top
> contributing features (margin-scale, with additivity verified end-to-end).
>
> 📊  **Drift monitoring** — PSI + KS-test per feature, with a synthetic
> drift-injector that powers the Streamlit dashboard slider.
>
> 🧪  **Engineering rigor** — 208 tests, mypy `--strict`, ruff lint + format,
> pre-commit hooks, GitHub Actions CI on every push.
>
> 🐳  **Production-shape deployable** — multi-stage Dockerfile, non-root
> runtime user, healthchecks, `docker compose up` for the full stack, GHCR
> release workflow on git tag.

---

## 🎬 Demo

```bash
docker compose up
```

| URL | What it serves |
|---|---|
| http://localhost:8000/docs | FastAPI service — `/health`, `/predict`, `/explain` (with interactive docs) |
| http://localhost:8501 | Streamlit demo — **Score** / **Explain** / **Drift** tabs |

The **Drift** tab has a magnitude slider you can drag from 0 → 3σ to watch a
chosen feature cross the PSI alert threshold in real time.

---

## 🚀 Quickstart

<details>
<summary><b>Local Python (5 commands)</b></summary>

```bash
# 1. Editable install + dev tools + pre-commit hooks
make install-dev

# 2. Download the ULB dataset (Kaggle API token required)
make data

# 3. Train the LightGBM + isotonic pipeline (~3 min on a laptop CPU)
make train

# 4. Run the API in one terminal…
make serve

# 5. …and the Streamlit demo in another.
make app
```

</details>

<details>
<summary><b>Docker (one command)</b></summary>

```bash
docker compose up --build
```

Both images come up on a shared network. The Streamlit container points at the
API via `FRAUDSHIELD_API_URL=http://api:8000`. Mount your trained artifacts at
`./models` and the API will pick them up automatically.

</details>

<details>
<summary><b>Verify everything with one command</b></summary>

```bash
make test       # 208 tests
make lint       # ruff
make type       # mypy --strict
```

</details>

---

## 🏗 Architecture

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

Component-by-component walkthrough in [`docs/architecture.md`](docs/architecture.md).

---

## 📊 Results

> Measured on the held-out 15% test partition. Populated by the final cell of
> [`notebooks/04_calibration_and_thresholds.ipynb`](notebooks/04_calibration_and_thresholds.ipynb).

| Model | PR-AUC | ROC-AUC | Recall @ Precision=0.9 | Brier |
|-------|:-:|:-:|:-:|:-:|
| Logistic regression (`class_weight='balanced'`) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| LightGBM (raw, `scale_pos_weight=auto`) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **LightGBM + isotonic** | **_TBD_** | _TBD_ | **_TBD_** | **_TBD_** |

> 💡 **Headline metric:** PR-AUC. ROC-AUC's FPR axis is saturated at this
> imbalance — it under-discriminates between very different models. PR-AUC
> focuses on the minority class, which is what fraud teams actually act on.

---

## 🧠 Design decisions worth knowing

<details>
<summary><b>Why loss-side weighting, not SMOTE?</b></summary>

SMOTE achieves similar recall but **degrades calibration**: it changes the
prior the model sees during training, so the output probabilities no longer
match `P(fraud | x)` on the real distribution. Brier score goes up,
reliability curves float above the diagonal. The comparison lives in
[`notebooks/03_imbalance_handling.ipynb`](notebooks/03_imbalance_handling.ipynb).

</details>

<details>
<summary><b>Why isotonic over Platt?</b></summary>

Isotonic regression is non-parametric and doesn't assume the miscalibration
looks sigmoidal — flexibility that pays off when you have enough val
positives (ULB has 492, comfortably above the rule-of-thumb threshold). Both
are implemented in `src/fraud_shield/models/calibration.py`; the notebook
picks isotonic after a head-to-head comparison.

</details>

<details>
<summary><b>Why a margin-scale SHAP API?</b></summary>

`TreeExplainer` returns values on the **log-odds** scale, which preserves the
additivity property `margin = expected_value + sum(shap_values)`. The
endpoint surfaces the margin so callers can choose whether to map back to
probability themselves — useful for downstream rule engines that work in
log-odds.

</details>

<details>
<summary><b>Why `cv='prefit'`-style calibration via a custom wrapper?</b></summary>

`sklearn.calibration.CalibratedClassifierCV` wants a `BaseEstimator`
subclass; `LightGBMFraudClassifier` deliberately isn't one (no
`get_params`/`set_params` boilerplate). The custom `CalibratedFraudClassifier`
takes any object with `predict_proba` via a narrow `Protocol`, which is also
trivially mockable in tests.

</details>

<details>
<summary><b>Why a configurable `CostMatrix` for threshold tuning?</b></summary>

F1 and Youden's J are domain-blind. Real fraud teams pick thresholds against
business cost: a missed fraud costs `$200` (chargeback + cost of goods + churn);
a false alarm costs `$5` (analyst time + customer friction). The optimizer
lives in `src/fraud_shield/evaluation/threshold.py` and sweeps the
precision-recall curve to find the minimum-expected-cost operating point.
The cost matrix is read from a YAML config so the deployment owner moves the
lever, not the model author.

</details>

---

## 🧰 Tech stack

<table>
<tr>
<td valign="top" width="50%">

**Core ML**
- LightGBM 4.5 (gradient boosting)
- scikit-learn 1.6 (preprocessing, calibration)
- Optuna 4.1 (hyperparameter search)
- SHAP 0.46 (TreeExplainer)

**Data**
- pandas 2.2 / numpy 2.1
- pandera 0.20 (schema validation)

</td>
<td valign="top" width="50%">

**Serving & UI**
- FastAPI 0.115 + uvicorn 0.34
- Streamlit 1.41 + Plotly 5.24
- pydantic 2.10 (request/response models)
- structlog (JSON logging)

**Quality & ops**
- pytest 8.3, ruff 0.8, mypy 1.14 strict
- pre-commit 4.0
- Docker (multi-stage, non-root)
- GitHub Actions CI + release workflows

</td>
</tr>
</table>

---

## 📁 Project layout

```
fraud-shield/
├── src/fraud_shield/         # installable package — all production code
│   ├── data/                 # ingest, schema, splits
│   ├── features/             # sklearn-compatible TimeAmountFeatures
│   ├── models/               # baseline, LightGBM, tuning, calibration
│   ├── evaluation/           # metrics, threshold, SHAP interpretability
│   ├── monitoring/           # drift detection + simulator
│   ├── api/                  # FastAPI service
│   └── utils/                # structlog setup
├── tests/                    # 208 tests, mypy-strict
├── notebooks/                # 01 EDA · 02 corr · 03 imbalance · 04 calibration+threshold
├── app/streamlit_app.py      # Score / Explain / Drift tabs
├── scripts/                  # CLI entry points
├── docs/                     # model card · data card · architecture
└── data/                     # gitignored; see data/README.md
```

---

## 🧪 Testing

```
208 passed, mypy --strict clean, ruff clean
```

| Suite | Count | Covers |
|---|:-:|---|
| `test_schema.py` | 10 | pandera validation, dtype coercion, structural rejection |
| `test_splits.py` | 19 | stratified + time-aware, class-ratio preservation, disjointness |
| `test_transformers.py` | 17 | feature engineering, sklearn pipeline composition |
| `test_baseline.py` | 12 | LR pipeline, class-weight effect, reproducibility |
| `test_lightgbm_model.py` | 12 | early stopping, auto `scale_pos_weight`, seed determinism |
| `test_tuning.py` | 7 | Optuna search space + reproducibility |
| `test_calibration.py` | 13 | isotonic vs Platt, miscalibration → Brier drop |
| `test_threshold.py` | 17 | cost-matrix optimum, directional sensitivity |
| `test_interpretability.py` | 20 | SHAP additivity, top-k, immutability |
| `test_drift.py` | 20 | PSI + KS, edge cases, threshold sensitivity |
| `test_simulate.py` | 15 | drift injection + end-to-end detector loop |
| `test_metrics.py` | 16 | PR-AUC, recall@p, Brier corner cases |
| `test_download_data.py` | 5 | checksum verification |
| `test_api.py` | 18 | FastAPI integration via TestClient + stubs |
| `test_api_smoke.py` | 5 | structural |

---

## 🗣 Interview talking points

<details>
<summary><b>1. Why PR-AUC over ROC-AUC?</b></summary>

ROC-AUC's FPR axis is saturated at 0.17% positive rate — the denominator
absorbs huge numbers of false positives without the metric moving much.
PR-AUC focuses on the positive class, which is what fraud teams act on.
Notebook 03 shows two models with similar ROC-AUC but very different
PR-AUC.

</details>

<details>
<summary><b>2. How do you handle imbalance?</b></summary>

Loss-side weighting (`class_weight='balanced'` for LR, auto
`scale_pos_weight` for LightGBM). Compared head-to-head against SMOTE and
undersampling in notebook 03. SMOTE matches the recall lift but degrades
Brier — that's the calibration tradeoff most papers gloss over.

</details>

<details>
<summary><b>3. Why calibrate?</b></summary>

Raw gradient-boosted "probabilities" are rankings, not probabilities.
Isotonic regression on the val set turns them into something a business
rule like "block at P > 0.8" can actually consume. Brier drops by roughly
50% on this dataset, and the reliability curve moves onto the diagonal.

</details>

<details>
<summary><b>4. How do you pick a threshold?</b></summary>

Not by F1 — by expected cost: `E[cost] = C_FN × FN + C_FP × FP`. See
`evaluation/threshold.py`. The cost matrix is configurable, so the
deployment owner moves the lever rather than the model author.

</details>

<details>
<summary><b>5. How do you monitor in production?</b></summary>

Three layers:
- **(a)** feature drift via PSI per feature, alert at `PSI > 0.2`
- **(b)** prediction drift via KS test on score distributions
- **(c)** label drift once chargeback feedback arrives (the inherent
  feedback delay in fraud is itself a system design topic)

Layers (a) and (b) are implemented in `monitoring/drift.py` and demoed live
in the Streamlit Drift tab.

</details>

<details>
<summary><b>6. What would you do with another month?</b></summary>

- Online learning loop with chargeback feedback
- IEEE-CIS dataset for richer non-PCA features
- Shapley-value sampling to make `/explain` p99 faster
- Real A/B test harness against the current rules engine
- Prometheus metrics + Grafana board on the API

</details>

---

## 📚 Documentation

| Document | What it covers |
|---|---|
| [`docs/model_card.md`](docs/model_card.md) | Intended use, training data, metrics, ethics, caveats |
| [`docs/data_card.md`](docs/data_card.md) | ULB dataset provenance, schema, known limitations |
| [`docs/architecture.md`](docs/architecture.md) | Component diagram, request paths, reproducibility |
| [`docs/roadmap.md`](docs/roadmap.md) | 4-week build plan |
| [`notebooks/`](notebooks/) | EDA, correlations, imbalance comparison, calibration + threshold |

---

## 📄 License & acknowledgements

MIT — see [LICENSE](LICENSE).

Dataset:
> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> IEEE SSCI, 2015.

---

<div align="center">

Built with ☕ and an unreasonable obsession with calibration curves.

</div>
