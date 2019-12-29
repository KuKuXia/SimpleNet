"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np
from network.layers import Linear, Tanh
from network.nn import NeuralNet
from network.train import train

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = np.round(net.forward(x), 2)

    print(x, predicted, y)
