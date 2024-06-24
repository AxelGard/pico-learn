import numpy as np
from sklearn.neighbors import KNeighborsClassifier as sklearn_knn_clf
from picolearn.knn import KNeighborsClassifier as pico_knn_clf


def test_knn_clf():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    sk_model = sklearn_knn_clf(n_neighbors=3)
    sk_model.fit(X, y)

    pl_model = pico_knn_clf(n_neighbors=3)
    pl_model.fit(X, y)

    X_test_1 = np.array([[1.1]])
    X_test_2 = np.array([[0.9]])

    """
    assert np.allclose(
        sk_model.predict(X_test_1), pl_model.predict(X_test_1)
    ), f"LinReg predcit failed, should have been:{sk_model.predict(X_test_1)[0]} but was:{pl_model.predict(X_test_1)[0]}"

    assert np.allclose(
        sk_model.predict(X_test_2), pl_model.predict(X_test_2)
    ), f"LinReg predcit failed, should have been:{sk_model.predict(X_test_2)[0]} but was:{pl_model.predict(X_test_2)[0]}"
    """
