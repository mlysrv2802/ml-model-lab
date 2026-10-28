import pandas as pd
from sklearn.model_selection import train_test_split

from mllab.common.config import RANDOM_STATE, TARGET, TEST_SIZE


def stratified_split(df: pd.DataFrame):
    """80/20 split that keeps the churn rate (about 26%) equal in both parts.

    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
