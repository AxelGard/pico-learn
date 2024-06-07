import numpy as np
from sklearn.svm import LinearSVC as SKLearnSVC
from picolearn.svm import SVC as PicoLearnSVC


def test_svc():

    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    X_test = np.array([[-0.8, -1]])

    sk_model = SKLearnSVC()
    sk_model.fit(X, y)

    pl_model = PicoLearnSVC()
    pl_model.fit(X, y)

    print("SKlearn:", sk_model.coef_, sk_model.intercept_)
    print("Pico:", pl_model.coef_, pl_model.intercept_)

    assert np.allclose(
        sk_model.predict(X_test)[0], pl_model.predict(X_test)[0]
    ), f"SVC predcit failed, should have been:{sk_model.predict(X_test)[0]} but was:{pl_model.predict(X_test)[0]}"
    """assert np.allclose(
        sk_model.coef_, pl_model.coef_
    ), f"SVC coefficans failed, should have been:{sk_model.coef_} but was:{pl_model.coef_}"
    assert np.allclose(
        sk_model.intercept_, pl_model.intercept_
    ), f"SVC intercept failed, should have been:{sk_model.intercept_} but was:{pl_model.intercept_}"
    """
