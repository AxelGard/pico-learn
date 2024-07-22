import numpy as np

sigmoid = lambda x: 1 / (1 + np.e ** (-1 * x))
relu = lambda x: [x * 0.01, x][int(x > 0)]

sigmoid_derivative = lambda x: x * (1 - x)


class MLPClassifier:

    def __init__(
        self,
        max_iter: int = 300,
        random_state: int = 123,
        lr: float = 1e-3,
        solver: str = "sgd",
    ) -> None:
        self.max_iter = max_iter
        self.random_state = random_state
        self.lr = lr
        self.weights

    def fit(self, X: np.ndarray, y: np.ndarray):

        return self
