from mllab.common.data import load_churn
from mllab.common.preprocessing import make_preprocessor
from mllab.common.splits import stratified_split


def test_load_churn_is_clean_and_numeric_where_it_should_be():
    df = load_churn()
    assert df.shape == (7043, 20)
    assert df.isna().sum().sum() == 0
    assert "customerID" not in df.columns
    assert df["TotalCharges"].dtype == "float64"
    assert set(df["Churn"].unique()) == {0, 1}


def test_split_is_stratified_and_complete():
    X_train, X_test, y_train, y_test = stratified_split(load_churn())
    assert len(X_train) + len(X_test) == 7043
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert "Churn" not in X_train.columns


def test_scale_false_keeps_real_units_and_scale_true_standardises():
    X_train, *_ = stratified_split(load_churn())

    raw = make_preprocessor(scale=False)
    out = raw.fit_transform(X_train)
    tenure = out[:, list(raw.get_feature_names_out()).index("num__tenure")]
    assert tenure.max() == 72  # months, untouched

    scaled = make_preprocessor(scale=True)
    out = scaled.fit_transform(X_train)
    tenure = out[:, list(scaled.get_feature_names_out()).index("num__tenure")]
    assert abs(tenure.mean()) < 1e-6 and abs(tenure.std() - 1) < 1e-6
