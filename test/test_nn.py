from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier as sk_MLPClassifier
#from picolearn.nn import NN as pl_MLPClassifier
from sklearn.metrics import accuracy_score


def test_nn_mlp_clf():

    X, y = make_classification(n_samples=100, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1
    )
    print(X)

    sk_clf = sk_MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    y_pred = sk_clf.predict(X_test)
    print(y_pred)
    print(accuracy_score(y_test, y_pred))
    assert True


# test_nn_mlp_clf()
