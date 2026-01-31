# Data Card — ULB Credit Card Fraud Detection

## Source

- **Dataset:** ULB Credit Card Fraud Detection
- **Kaggle:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Original publication:** Dal Pozzolo et al., IEEE SSCI 2015
  ([PDF](https://www.researchgate.net/publication/283349138))
- **License:** Open Database License (ODbL)
- **Owner:** Worldline + Machine Learning Group, Université Libre de Bruxelles (ULB)

## Composition

- **Rows:** 284,807 transactions
- **Period:** two consecutive days in September 2013
- **Geography:** European cardholders
- **Class balance:**
  - 284,315 non-fraud (99.828%)
  - 492 fraud (0.172%)
- **Columns:**
  - `Time` — seconds since the first transaction in the dataset
  - `V1`..`V28` — principal components from a PCA transform applied
    by the dataset authors. The original features are withheld for
    privacy.
  - `Amount` — transaction amount in the local currency
  - `Class` — binary label (1 = fraud, 0 = legitimate)
- **File:** `creditcard.csv` (~144 MB uncompressed)
- **Encoding:** UTF-8, comma-separated, header in row 1

## How we use it

1. Validated against `fraud_shield.data.schema.TRANSACTION_SCHEMA` (pandera).
   The schema enforces dtypes, non-negativity on `Time` and `Amount`,
   and `Class ∈ {0, 1}`. Unknown columns are rejected.
2. Split via `fraud_shield.data.splits.stratified_random_split` —
   65/15/20 train/val/test with stratification on `Class` so all three
   partitions preserve the ~0.17% positive rate.
3. A time-aware split (`time_aware_split`) is also produced as a
   robustness check: training on the earliest transactions and testing
   on the latest reveals any temporal sensitivity the stratified split
   masks.

## Known limitations

- **PCA opacity.** The V features can't be inspected semantically, so
  feature engineering is limited to `Time` and `Amount`.
- **Class imbalance is on the high side.** Production fraud rates are
  typically 0.01–0.05% (an order of magnitude lower). Apparent metrics
  on this dataset may flatter the model.
- **Two-day window.** The dataset doesn't capture day-of-week, weekly
  seasonality, or holiday spikes — three signals a real fraud team
  would rely on.
- **No identity features.** No merchant ID, card BIN, geography, or
  device fingerprint. Real-world fraud models lean heavily on these.
- **Sampling.** The ULB team may have under-sampled negatives before
  publishing. Notebook 02 sanity-checks that the V features remain
  near-orthogonal (PCA was preserved through any post-processing).

## How to fetch

The CSV is not committed to this repo (gitignored — 144 MB). Each
developer pulls it locally:

```bash
make data
# or:
python -m scripts.download_data
```

Requires the Kaggle CLI on PATH (`pip install kaggle`) and an API
token at `~/.kaggle/kaggle.json` (Linux/macOS) or
`%USERPROFILE%\.kaggle\kaggle.json` (Windows). See `data/README.md`
for token instructions.

The download script verifies a SHA-256 checksum against the file
(once the constant is populated after the first verified download).

## Citation

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca
> Bontempi. *Calibrating Probability with Undersampling for Unbalanced
> Classification.* IEEE Symposium Series on Computational Intelligence
> (SSCI), 2015.
