import numpy as np
from sklearn.svm import LinearSVC as SKLearnSVC
from picolearn.svm import LinearSVC as PicoLearnSVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def test_svc():

    X, y = make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    sk_model = SKLearnSVC()
    sk_model.fit(X_train, y_train)

    pl_model = PicoLearnSVC()
    pl_model.fit(X_train, y_train)

    assert np.allclose(
        sk_model.predict(X_test)[0], pl_model.predict(X_test)[0]
    ), f"SVC predcit failed, should have been:{sk_model.predict(X_test)[0]} but was:{pl_model.predict(X_test)[0]}"
