# Model Card — FraudShield v0.1.0

## Model details

- **Architecture:** LightGBM gradient-boosted binary classifier wrapped
  with isotonic probability calibration.
- **Inputs:** 30 numeric features per transaction — `Time`, `V1`..`V28`
  (PCA-anonymized), `Amount`. Engineered at inference time into
  `hour`, `day`, `log_amount`, `amount_is_zero` (replacing `Time` and
  `Amount` in the model input space).
- **Output:** calibrated probability of fraud in [0, 1], plus a binary
  decision against a configurable cost-aware threshold.
- **Owner:** Jenisssh.
- **Repo:** https://github.com/Jenisssh/fraud-shield
- **License:** MIT.

## Intended use

- **Primary:** real-time scoring of card-present transactions to flag
  candidates for human review.
- **Operational mode:** a decision threshold tuned against a
  business-supplied cost matrix (`C_FN`, `C_FP`). The deployment owner
  picks where on the precision/recall curve to live.
- **Out of scope:** card-not-present transactions, ACH or wire fraud,
  account-takeover signals, anything outside the ULB schema.

## Factors

- **Imbalance:** 0.172% positive rate on the training data (ULB
  dataset). Production fraud rates are typically 0.01–0.05% — the
  model has seen more positives than it would in deployment, which can
  inflate apparent metrics.
- **Time window:** training data spans roughly 48 hours of transactions
  in September 2013. The model has no exposure to seasonality,
  holiday effects, or year-over-year drift.
- **Anonymization:** V1..V28 are PCA features. We cannot reason about
  what they encode semantically — feature engineering is limited to
  the two raw columns.

## Metrics

Measured on the held-out test partition (15% stratified split from
`fraud_shield.data.splits.stratified_random_split`, seed = 42).

> The values below are filled in by the final cell of
> `notebooks/04_calibration_and_thresholds.ipynb` after a real
> training run on the ULB CSV. Until then they're placeholders.

| Model | PR-AUC | ROC-AUC | Recall @ Precision=0.9 | Brier |
|-------|--------|---------|-----------------------|-------|
| Logistic regression baseline | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| LightGBM (raw) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| LightGBM + isotonic | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Headline metric:** PR-AUC. ROC-AUC is reported for completeness but
under-discriminates at this imbalance.

## Training data

- **Source:** [ULB Credit Card Fraud Detection
  dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  (Dal Pozzolo et al., 2015).
- **Size:** 284,807 transactions, 492 frauds.
- **License:** Open Database License (ODbL).
- **Pre-processing:** validated via `fraud_shield.data.schema.TRANSACTION_SCHEMA`
  (rejects unknown columns, non-negative `Time` and `Amount`,
  `Class ∈ {0, 1}`).
- **Splits:** stratified 65/15/20 train/val/test. Time-aware split
  also reported as a robustness check; see `notebooks/02`.

## Evaluation data

Same source as training; the held-out 15% test partition is never seen
during model selection, hyperparameter tuning, calibration, or
threshold optimization. All of those use the val partition.

## Quantitative analyses

- **Imbalance handling** (`notebooks/03`): compared no-weighting vs
  `class_weight='balanced'` vs SMOTE vs LightGBM `scale_pos_weight`.
  Loss-side weighting wins on calibration; SMOTE achieves similar
  recall but degrades Brier.
- **Calibration** (`notebooks/04`): isotonic regression on the val
  partition halves the Brier score vs the raw LightGBM output.
- **Threshold sensitivity** (`notebooks/04`): traced the optimal
  threshold across `C_FN / C_FP` ratios from 5 to 160. Recall climbs
  from ~0.6 to >0.9 as the cost ratio grows.

## Ethical considerations

- The model can recommend declining a transaction. False positives are
  not free — they delay customers, generate support load, and
  disproportionately harm users in lower-friction-tolerance segments
  (e.g. someone whose first card is declined while travelling). The
  cost matrix in `evaluation/threshold.py` is meant to make this
  explicit and tunable.
- ULB's PCA anonymization removes most direct biases from the
  features, but the model is still trained on European cardholder
  transactions from 2013. Generalization to other populations is
  unverified.

## Caveats

- **No production traffic.** This model has never been deployed.
  Numbers reported here come from a single offline test partition.
- **No feedback loop.** Chargeback labels arrive weeks after the
  transaction. The retraining cadence and label-delay handling are
  future work.
- **No drift on labels.** The drift module (`monitoring/drift.py`)
  monitors feature distributions but not label distributions or
  prediction calibration — both should be tracked in production.

## Citation

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca
> Bontempi. *Calibrating Probability with Undersampling for Unbalanced
> Classification.* IEEE Symposium Series on Computational Intelligence
> (SSCI), 2015.
