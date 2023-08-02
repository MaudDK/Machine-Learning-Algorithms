import numpy as np
import tensorflow as tf

class CollaborativeFiltering:
    def __init__(self, seed=42):
        tf.randoms.set_seed(seed)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    def fit(self, X, Y, R, max_iters, lambda_):
        W = 0.1 * np.random.randn((Y.shape[1], X.Shape[1]))
        b = np.zeros((1, Y.shape[1]))
        
        for _ in range(max_iters):
            with tf.GradientTape() as tape:
                cost = self.cost_function(X, W, b, Y, R, lambda_)
        
        grads = tape.gradient(cost, [X,W,b])
        self.optimizer.apply_gradients(zip(grads, [X,W,b]))

        if iter % 20 == 0:
            print(f"Training loss at iteration {iter}: {cost:0.1f}")

    def cost_function(self, X, W, b, Y, R, lambda_):
        j = np.sum(np.square((np.dot(X, W.T)+b - Y) * R))/2
        reg = (lambda_/2) * (np.sum(np.square(W)) + np.sum(np.square(X)))
        J = j + reg
        return J