from pathlib import Path

import pandas as pd

from mllab.common.config import CHURN_PATH, ID_COLUMN, TARGET


def load_churn(path: Path = CHURN_PATH) -> pd.DataFrame:
    """Telco churn data, cleaned and ready for modelling.

    - TotalCharges is text in the CSV (11 blanks for brand-new customers), so it
      becomes a number, with blanks as 0.
    - Churn becomes 1 (left) / 0 (stayed).
    - customerID is dropped: it is unique per row, so it cannot generalise.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df.drop(columns=[ID_COLUMN])
