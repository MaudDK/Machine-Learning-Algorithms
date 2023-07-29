import numpy as np
np.random.seed(0)
from models.neural_networks.activation.Activation import Activation

class Dense:
    def __init__(self, n_inputs, units, activation = None):
        self.weights = 0.1 * np.random.randn(units, n_inputs)
        self.bias = np.zeros((units, 1))
        self.type = "Dense"
        self.params = units * (n_inputs + 1)
        self.activation = Activation(activation)

    def forward(self, inputs):
        print("Input Shape:", inputs.shape)
        print("Weights Shape:", self.weights.shape)
        print("Bias Shape:", self.bias.shape)
        print("Output Shape:", (self.weights.shape[0], inputs.shape[1]), "\n")
        self.output = np.dot(self.weights, inputs) + self.bias
        #self.output = np.matmul(self.weights, inputs) + self.bias
        self.output = self.activation.forward(self.output)
        return self.output
