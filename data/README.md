# Data

The ULB Credit Card Fraud Detection dataset is **not** committed to this repo
(see `.gitignore`). Each developer pulls it locally before training.

## What you need

- A free Kaggle account: https://www.kaggle.com/account
- A Kaggle API token saved at `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows).
  Generate it under *Account → API → Create New Token*.

## How to fetch

```bash
make data
```

or manually:

```bash
python -m scripts.download_data
```

This downloads `creditcard.csv` (~144 MB) into `data/raw/` and verifies its
SHA-256 checksum.

## Dataset structure

| Column | Description |
|--------|-------------|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1`–`V28` | PCA-transformed features (original columns withheld for privacy) |
| `Amount` | Transaction amount |
| `Class` | Target — `1` for fraud, `0` otherwise |

- **Rows:** 284,807
- **Frauds:** 492 (0.172%)
- **Time span:** 2 days of European cardholder transactions, September 2013
- **License:** Open Database License (ODbL) — credit Worldline + ULB MLG

## Citation

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi.
*Calibrating Probability with Undersampling for Unbalanced Classification.*
In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

## Layout

```
data/
├── raw/         # creditcard.csv lands here — gitignored
├── interim/     # any intermediate artifacts (cached splits, etc.)
├── processed/   # final train/val/test parquets
└── README.md    # this file
```

A `.gitkeep` file in each subdirectory preserves the structure in git.
