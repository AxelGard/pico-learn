import numpy as np


class SVC:
    """Support Vector Machine Classifier"""

    def __init__(self, kernal="linear") -> None:
        self.coef_: np.ndarray = None  # noted as w

    def fit(self, X: np.ndarray, y: np.ndarray):
        b: np.ndarray = None
        w: np.ndarray = None
        w = b

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
