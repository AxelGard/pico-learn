import numpy as np
from sklearn.svm import SVC as SKLearnSVC
from picolearn.svm import SVC as PicoLearnSVC


def test_svc():
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    X_test = np.array([[-0.8, -1]])

    sk_model = SKLearnSVC(kernel="linear")
    sk_model.fit(X, y)

    print(sk_model.coef_)

    y_pred = sk_model.predict(X_test)
    print(y_pred)

    print("-" * 10)

    pl_model = PicoLearnSVC()
    pl_model.fit(X, y)
    y_pred = pl_model.predict(X_test)
    print(y_pred)
