from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier as sk_MLPClassifier
from picolearn.nn import NN as pl_MLPClassifier


def test_nn_mlp_clf():

    X, y = make_classification(n_samples=100, random_state=1)
    X_train, X_test, y_train, _ = train_test_split(X, y, stratify=y, random_state=1)

    sk_clf = sk_MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    sk_clf.predict_proba(X_test[:1])
