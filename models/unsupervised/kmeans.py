import numpy as np

class KMeans:
    def __init__(self, k = 3):
        self.k = k
        self.centroids = None

    def fit(self, X, max_iters = 1000):
        #Random Initialization of Centroids
        self.centroids = self.random_centroids_init(X)

        for i in range(max_iters):
            print(f"K-Means iteration {i}/{max_iters-1}")
            y = self.predict(X)
            self.update_centroids(X, y)

    
    def random_centroids_init(self, X):
        randidx = np.random.permutation(X.shape[0])
        centroids = X[randidx[:self.k]]
        return centroids
    
    def predict(self, X):
        y = np.zeros(X.shape[0], dtype = int)

        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - self.centroids, axis= 1)
            y[i] = np.argmin(distances)
        
        return y
    
    def update_centroids(self, X, y):
        for k in range(self.k):
            points = X[y == k]
            self.centroids[k] = np.mean(points, axis = 0)
