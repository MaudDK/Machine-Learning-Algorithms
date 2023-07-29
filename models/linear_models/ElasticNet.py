import numpy as np

class ElasticNet:
    def __init__(self, n_iters = 1000, learning_rate = 0.01, l1 = 0.01, l2 = 0.01):
        self.n_iters = n_iters
        self.alpha = learning_rate
        self.l1 = l1
        self.l2 = l2
    
    def fit(self, X, y):
        m, j = X.shape
        self.W = 0.10 * np.random.rand(j)
        self.b = 0

        for _ in range(self.n_iters):
            y_pred = self.predict(X)

            dw = 1/m * np.dot(X.T, y_pred - y) + self.l1/m + (1/m) * self.l2 * np.sum(self.W)
            db =  np.mean(y_pred - y)

            self.W = self.W - self.alpha * dw
            self.b = self.b - self.alpha * db

    def predict(self, X):
        return np.dot(self.W, X.T) + self.b