import numpy as np
from sklearn.linear_model import LinearRegression as SKLearnLinearRegression
from picolearn.linear import LinearRegrasion as PicoLearnLinearRegression
from sklearn.datasets import make_regression


def test_linreg():

    X, y = make_regression(n_samples=200, n_features=10, random_state=1)

    X_test, _ = make_regression(n_samples=5, n_features=10, random_state=2)

    sk_model = SKLearnLinearRegression()
    sk_model.fit(X, y)

    pl_model = PicoLearnLinearRegression()
    pl_model.fit(X, y)

    assert np.allclose(
        sk_model.predict(X_test), pl_model.predict(X_test)
    ), f"LinReg predcit failed, should have been:{sk_model.predict(X_test)[0]} but was:{pl_model.predict(X_test)[0]}"
    assert np.allclose(
        sk_model.coef_, pl_model.coef_
    ), f"LinReg coefficans failed, should have been:{sk_model.coef_} but was:{pl_model.coef_}"
    assert np.allclose(
        sk_model.intercept_, pl_model.intercept_
    ), f"LinReg intercept failed, should have been:{sk_model.intercept_} but was:{pl_model.intercept_}"
