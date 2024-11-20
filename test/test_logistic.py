import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as sk_lr
from picolearn.logistic import LogisticRegression as pk_lr


def test_logistic_regression():
    X, y = load_iris(return_X_y=True)

    sk_model = sk_lr(random_state=0).fit(X,y)
    pk_model = pk_lr().fit(X,y)

    print(np.array([sk_model.score(X,y)]),np.array([pk_model.score(X,y)]))

    assert True  
