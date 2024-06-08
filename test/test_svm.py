import numpy as np
from sklearn.svm import LinearSVC as SKLearnSVC
from picolearn.svm import LinearSVC as PicoLearnSVC
from sklearn.datasets import make_regression, make_classification


def test_svc():

    X, y = make_classification(n_samples=200, random_state=1)
    X_test, _ = make_classification(n_samples=5, random_state=2)

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
