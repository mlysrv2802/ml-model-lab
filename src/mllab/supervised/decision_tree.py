import pandas as pd
from sklearn.model_selection import StratifiedKFold, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from mllab.common.config import RANDOM_STATE
from mllab.common.preprocessing import make_preprocessor

DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 25]


def build_decision_tree(max_depth: int | None = None) -> Pipeline:
    """A single decision tree; max_depth=None lets it grow until it memorises the data.

    Trees compare one column against a threshold at a time, so scaling changes
    nothing. We skip it, and the learned rules stay in real units (months, dollars).
    """
    return Pipeline(
        [
            ("prep", make_preprocessor(scale=False)),
            ("model", DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)),
        ]
    )


def depth_curve(X, y, depths=DEFAULT_DEPTHS, folds: int = 5) -> pd.DataFrame:
    """ROC-AUC on the data the tree trained on vs on data it never saw, per max_depth.

    The widening gap between the two columns is overfitting made visible.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    train, validation = validation_curve(
        build_decision_tree(),
        X,
        y,
        param_name="model__max_depth",
        param_range=depths,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    out = pd.DataFrame(
        {"train_auc": train.mean(axis=1), "validation_auc": validation.mean(axis=1)},
        index=pd.Index(depths, name="max_depth"),
    )
    out["gap"] = out["train_auc"] - out["validation_auc"]
    return out


def tree_rules(X, y, max_depth: int = 3) -> str:
    """The learned if/else rules of a shallow tree, as readable text."""
    tree = build_decision_tree(max_depth=max_depth).fit(X, y)
    names = tree.named_steps["prep"].get_feature_names_out()
    return export_text(tree.named_steps["model"], feature_names=list(names))
