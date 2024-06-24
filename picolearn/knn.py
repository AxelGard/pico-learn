import numpy as np


class KNeighborsClassifier:

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        return X
