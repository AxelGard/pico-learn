import numpy as np
from picolearn.tree import DecisionTreeClassifier as PicoDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier


def test_decision_tree_classifier():
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    X_test = np.array([[-0.8, -1]])

    sk_model = SklearnDecisionTreeClassifier()
    sk_model.fit(X, y)

    pl_model = PicoDecisionTreeClassifier()
    pl_model.fit(X, y)

    print(sk_model.predict(X_test))
    print("-" * 10)
    print(pl_model.predict(X_test))
