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


class LinearSVC:
    def __init__(self, learning_rate=0.001, lamda_parm=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lamda_parm = lamda_parm
        self.epochs = epochs
        self.classes: np.ndarray = None
        self.coef_: np.ndarray = None
        self.intercept_: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray):

        _, nr_of_features = X.shape
        self.classes = np.unique(y)
        nr_of_classes = len(self.classes)

        self.coef_ = np.zeros((nr_of_classes, nr_of_features))
        self.intercept_ = np.zeros(nr_of_classes)

        for cls_idx, class_ in enumerate(self.classes):
            y_where_cls = np.where(y == class_, 1, -1)
            for _ in range(self.epochs):
                for i, x_i in enumerate(X):
                    if (
                        y_where_cls[i]
                        * (np.dot(x_i, self.coef_[cls_idx]) - self.intercept_[cls_idx])
                        >= 1
                    ):
                        self.coef_[cls_idx] -= self.learning_rate * (
                            2 * self.lamda_parm * self.coef_[cls_idx]
                        )
                    else:
                        self.coef_[cls_idx] -= self.learning_rate * (
                            2 * self.lamda_parm * self.coef_[cls_idx]
                            - np.dot(x_i, y_where_cls[i])
                        )
                        self.intercept_[cls_idx] -= self.learning_rate * y_where_cls[i]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_outputs = np.dot(X, self.coef_.T) - self.intercept_
        print(linear_outputs)
        return self.classes[np.argmax(linear_outputs, axis=1)]
