import pytest

from mllab.common.data import load_churn
from mllab.common.splits import stratified_split
from mllab.supervised.decision_tree import build_decision_tree, depth_curve, tree_rules


@pytest.fixture(scope="module")
def train():
    X_train, _, y_train, _ = stratified_split(load_churn())
    return X_train, y_train


def test_max_depth_limits_the_tree(train):
    X, y = train
    tree = build_decision_tree(max_depth=3).fit(X, y)
    assert tree.named_steps["model"].get_depth() <= 3


def test_unlimited_tree_memorises_the_training_data(train):
    X, y = train
    curve = depth_curve(X, y, depths=[1, 5, 25], folds=3)
    assert curve.loc[25, "train_auc"] > 0.99          # memorised
    assert curve.loc[25, "gap"] > 0.2                  # but fails on unseen rows
    assert curve.loc[1, "gap"] < 0.02                  # a stump cannot overfit
    assert curve.loc[5, "validation_auc"] > curve.loc[25, "validation_auc"]


def test_tree_rules_are_readable_text_in_real_units(train):
    X, y = train
    rules = tree_rules(X, y, max_depth=2)
    assert "class:" in rules
    assert "Contract" in rules
    assert "-1." not in rules  # no standardised (negative) tenure thresholds
