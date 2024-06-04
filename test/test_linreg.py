import numpy as np
from sklearn.linear_model import LinearRegression as SKLearnLinReg
from picolearn.linear import LinearRegrasion as PLLinReg


def test_linreg():

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    X_test = np.array([[3, 5]])

    sk_model = SKLearnLinReg()
    sk_model.fit(X, y)

    pl_model = PLLinReg()
    pl_model.fit(X, y)

    assert np.allclose(sk_model.predict(X_test)[0], pl_model.predict(X_test)[0])
    assert np.allclose(sk_model.coef_, pl_model.coef_)
    assert np.allclose(sk_model.intercept_, pl_model.intercept_)
