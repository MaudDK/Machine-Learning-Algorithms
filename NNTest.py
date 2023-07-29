from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from models.neural_networks.layers.Dense import Dense
from models.neural_networks.models.Sequential import Sequential


X = np.array([
     [ 1.0,  2.0,  3.0,  2.5],
     [ 2.0,  5.0, -1.0,  2.0],
     [-1.5,  2.7,  3.3, -0.8],
     ])

n_inputs, n_features = X.shape
layer_1 = Dense(n_inputs = n_inputs, units = 5, activation="relu")
layer_2 = Dense(n_inputs=5, units= 3, activation="relu")
layer_3 = Dense(n_inputs=3, units= 3, activation="softmax")

model = Sequential("Test Model")
model.add(layer_1)
model.add(layer_2)
model.add(layer_3)
model.summary()

model.predict(X)
for layer, output in enumerate(model.outputs):
    print(f"Layer {layer}:\n{output}\n")








