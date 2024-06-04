import numpy as np


class LinearRegrasion:

    def __init__(self) -> None:
        self.coef_: np.ndarray = None
        self.intercept_ = 0.0
        self.B: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """https://en.wikipedia.org/wiki/Linear_least_squares"""
        X_b = np.c_[np.ones(y.shape), X]
        self.B = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.coef_ = self.B[1:]
        self.intercept_ = self.B[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.c_[np.ones(X.shape[0]), X]
        y_pred = X_b.dot(self.B)
        return y_pred
