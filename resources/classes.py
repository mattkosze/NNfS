import numpy as np

class DenseLayer:
    def __init__(self, nInputs, nNeurons):
        # Method to randomly initialize the weights and biases
        self.weights = 0.01 * np.random.randn(nInputs, nNeurons) # creates matrix of size (nInputs x nNeurons) with random integers between -1 and 1 scaled down by 0.01. We scale this down to have weights a couple of magnitudes smaller, meaning less time spent fitting. Not an absolute rule, and something to play with.
        self.biases = np.zeros((1, nNeurons)) # creates an array filled with 0's of size (1 x nNeurons)

    def forward(self, inputs):
        # Calculates the output of a forward pass
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class SoftMax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_vals/np.sum(exp_vals, axis=1, keepdims=True)
        self.output = probabilities
