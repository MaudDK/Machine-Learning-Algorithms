import numpy as np

class LinearRegression:
    def __init__(self, n_iters = 1000, learning_rate = 0.01):
        self.n_iters = n_iters
        self.alpha = learning_rate
    
    def fit(self, X, y):
        m, j = X.shape
        self.W = 0.10 * np.random.rand(j)
        self.W = np.zeros(j)
        self.b = 0

        for _ in range(self.n_iters):
            y_pred = self.predict(X)

            dw = 1/m * np.dot(X.T, y_pred - y)
            db =  np.mean(y_pred - y)

            self.W = self.W - self.alpha * dw
            self.b = self.b - self.alpha * db

    def predict(self, X):
        return np.dot(self.W, X.T) + self.b