import numpy as np


class LinearSVC:
    """
    Linear Support Vector Machine Classifier
    src: https://youtu.be/T9UcK-TxQGw?si=5B1pKYBB3HP-jqmm
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000) -> None:
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = np.zeros(1)  # weights
        self.b = 0  # baies

    def fit(self, X: np.ndarray, y: np.ndarray):
        _, n_featrues = X.shape

        y_ = np.where(y <= 0, -1, 1)  # y values need to be 1 or -1

        self.w = np.zeros(n_featrues)  # random wights is better the 0
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                con = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if con:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

        return self

    def predict(self, X: np.ndarray):
        return np.sign(np.dot(X, self.w) - self.b)
