import numpy as np


class DecisionTreeClassifier:
    """A decision tree classifier"""

    def __init__(self, random_state=0) -> None:
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
