import numpy as np

class Activation:
    def __init__(self, activation = None):
        self.activation = activation

    def forward(self, X):
        if self.activation == "sigmoid":
            return self.sigmoid(X)

        elif self.activation == "relu":
            return self.relu(X)
        
        return X

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        return np.exp(Z)/np.sum(np.exp(Z))
