from mllab.common.data import load_churn
from mllab.common.splits import stratified_split
from mllab.supervised.decision_tree import depth_curve, tree_rules


def main() -> None:
    X_train, X_test, y_train, y_test = stratified_split(load_churn())

    print("1. The if/else rules a depth-3 tree learned (class 1 = churn)\n")
    print(tree_rules(X_train, y_train, max_depth=3))

    print("2. Same tree type, different max_depth (5-fold CV on the TRAIN set)")
    print("   train_auc = score on data it trained on; validation_auc = score on unseen data\n")
    print(depth_curve(X_train, y_train).round(3).to_string())


if __name__ == "__main__":
    main()
