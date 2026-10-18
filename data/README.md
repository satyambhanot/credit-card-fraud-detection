# Data

The dataset is not included in this repository (the CSV is about 144 MB).

1. Download `creditcard.csv` from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Put it at `data/raw/creditcard.csv`.

The file has 284,807 rows and 31 columns: `Time`, `V1` to `V28`, `Amount` and the label `Class` (1 for fraud, 0 otherwise). Only 492 rows are fraud.

`src/01_preprocess.py` writes scaled train/test splits to `data/processed/`, which is also not tracked.
