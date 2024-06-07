import numpy as np


class SVC:
    """Support Vector Machine Classifier"""

    def __init__(self, kernal="linear") -> None:
        self.coef_: np.ndarray = None  # noted as w
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=1000, learning_rate=1e-2):
        w = np.zeros(X.shape[1])
        b = 0
        for _ in range(epochs):
            for i in range(len(X)):
                x_i = X[i]
                y_i = y[i]
                if y_i * np.dot(x_i, w) + b < 1:
                    w += learning_rate * y_i * x_i
                    b += learning_rate * y_i

        self.coef_ = np.array([w])
        self.intercept_ = np.array([b])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(np.dot(X.T, self.coef_) + self.intercept_)
