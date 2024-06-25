import numpy as np


class KNeighborsClassifier:
    """src: https://youtu.be/0p0o5cmgLdE?si=TAiNCtKhXl1DnkWk"""

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors
        self.X_train = np.zeros(1)
        self.y_train = np.zeros(1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.zeros(X.shape)
        dist = np.linalg.norm(X[:, np.newaxis] - self.X_train, axis=2)  # Euclidean dist
        nearest_neighbors = np.argsort(dist, axis=1)[
            :, : self.n_neighbors
        ]  # get n neigbors to X
        for i, neighbors in enumerate(nearest_neighbors):
            neighbors_classifications = self.y_train[neighbors]
            # the most common class of neighbors i the new class for each point
            pred[i] = np.argmax(np.bincount(neighbors_classifications))
        return pred


class KNeighborsRegressor:
    """src: https://youtu.be/0p0o5cmgLdE?si=TAiNCtKhXl1DnkWk"""

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors
        self.X_train = np.zeros(1)
        self.y_train = np.zeros(1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.zeros(X.shape)
        dist = np.linalg.norm(X[:, np.newaxis] - self.X_train, axis=2)  # Euclidean dist
        nearest_neighbors = np.argsort(dist, axis=1)[
            :, : self.n_neighbors
        ]  # get n neigbors to X
        for i, neighbors in enumerate(nearest_neighbors):
            pred[i] = np.sum(self.y_train[neighbors]) / self.n_neighbors
        return pred
