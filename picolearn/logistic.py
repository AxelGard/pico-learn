import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000) -> None:
        """ logistic regression classifier 
        https://en.wikipedia.org/wiki/Logistic_regression """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = np.array([0]) 
        self.bias = 0 

    @staticmethod
    def sigmod(x:np.ndarray)->np.ndarray:
        return 1 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            lin_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmod(lin_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self

    def predict_prob(self, X:np.ndarray) -> np.ndarray:
        lin_model = np.dot(X, self.weights) + self.bias
        return self.sigmod(lin_model)

    def predict(self, X: np.ndarray, threshold:float=0.5) -> np.ndarray:
        return (self.predict_prob(X) >= self.bias).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float: 
        return np.mean(self.predict(X) == y)

