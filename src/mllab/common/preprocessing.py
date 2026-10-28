import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Turn a table of numbers and text into a pure-number table.

    - Text columns become 0/1 columns (one-hot encoding).
    - Number columns are standardised when scale=True. Models that measure
      distances or fit weights (kNN, SVM, logistic regression) need this;
      tree-based models do not, so they pass scale=False and keep real units.

    Column types are found when the preprocessor is fitted, so wrapped inside a
    Pipeline it only ever learns from training data (no leakage).
    """
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_exclude=np.number),
            ),
        ]
    )
