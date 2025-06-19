import numpy as np

class DenseLayer:
    def __init__(self, nInputs, nNeurons):
        # Method to randomly initialize the weights and biases
        self.weights = 0.01 * np.random.randn(nInputs, nNeurons) # creates matrix of size (nInputs x nNeurons) with random integers between -1 and 1 scaled down by 0.01. We scale this down to have weights a couple of magnitudes smaller, meaning less time spent fitting. Not an absolute rule, and something to play with.
        self.biases = np.zeros((1, nNeurons)) # creates an array filled with 0's of size (1 x nNeurons)

    def forward(self, inputs):
        # Calculates the output of a forward pass
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        # Gradients on parameters; use inputs because it's with respect to the weights
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on input values; we use weights because it's with respect to the inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        # Creates a copy of the dvalues array
        self.dinputs = dvalues.copy()
        # Iterates through the array and sets every value leq 0 to 0
        self.dinputs[self.inputs <= 0] = 0

class SoftMax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_vals/np.sum(exp_vals, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Creates uninitialized arr
        self.dinputs = np.empty_like(dvalues)

        # Enum. outputs and gradients
        for index, (singleOutput, singleDvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output arr
            singleOutput = singleOutput.reshape(-1,1)
            # Calculate jacobian matrix
            jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)
            # Sample-wise gradient calculation
            self.dinputs[index] = np.dot(jacobianMatrix, singleDvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CategoricalCrossEntropy(Loss):
    # number of samples in the batch
    def forward(self, y_pred, y_true):
        # identify the number of samples
        samples = len(y_pred)

        # clip data to prevent absolute 0's
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1: # If there are sparsely-encoded labels
            confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # If there are one-hot encoded labels
            confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # calculate losses
        losses = -np.log(confidences)
        return losses

    def backward(self, dvalues, y_true):
        # identify the number of samples
        samples = len(dvalues)

        # find the number of labels in every sample
        labels = len(dvalues[0])

        # transform to one-hot encoding if sparse
        if len(y_true.shape) == 1:
            # np.eye() creates a 2d array with ones on a specific diagonal and zeros elsewhere; an "a la identity" matrix
            y_true = np.eye(labels)[y_true]

        # calculate the gradient
        self.dinputs = -y_true / dvalues
        # normalize the gradient
        self.dinputs /= samples

class SoftMaxCategoricalCrossEntropy():
    # Creates new activation and loss function objects
    def __init__(self):
        self.activation = SoftMax()
        self.loss = CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layers activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Spare encoded guaranteed
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy the values for modification
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs /= samples