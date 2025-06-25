import numpy as np

class DenseLayer:
    def __init__(self, nInputs, nNeurons, weightl1=0, weightl2=0, biasl1=0, biasl2=0):
        # Method to randomly initialize the weights and biases
        self.weights = 0.01 * np.random.randn(nInputs, nNeurons) # creates matrix of size (nInputs x nNeurons) with random integers between -1 and 1 scaled down by 0.01. We scale this down to have weights a couple of magnitudes smaller, meaning less time spent fitting. Not an absolute rule, and something to play with.
        self.biases = np.zeros((1, nNeurons)) # creates an array filled with 0's of size (1 x nNeurons)
        # Store regularization strength
        self.weightl1 = weightl1
        self.weightl2 = weightl2
        self.biasl1 = biasl1
        self.biasl2 = biasl2

    def forward(self, inputs):
        # Calculates the output of a forward pass
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        # Gradients on parameters; use inputs because it's with respect to the weights
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L1 on weights
        if self.weightl1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.weightl1 * dl1
        # L1 on biases
        if self.biasl1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.biasl1 * dl1
        # L2 on weights
        if self.weightl2 > 0:
            self.dweights += 2 * self.weightl2 * self.weights
        # L2 on biases
        if self.biasl2 > 0:
            self.dbiases += 2 * self.biasl2 * self.biases

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

    def regularizationLoss(self, layer):
        # Set it to 0 by default
        regLoss = 0

        # L1 reg for weights
        if layer.weightl1 > 0:
            regLoss += layer.weightl1 * np.sum(np.abs(layer.weights))

        # L1 reg for biases
        if layer.biasl1 > 0:
            regLoss += layer.biasl1 * np.sum(np.abs(layer.biases))

        # L2 reg for weights
        if layer.weightl2 > 0:
            regLoss += layer.weightl2 * np.sum(layer.weights ** 2)

        # L2 reg for bises
        if layer.biasl2 > 0:
            regLoss += layer.biasl2 * np.sum(layer.biases ** 2)

        return regLoss

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

class SoftMaxCategoricalCrossEntropy:
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

class SGD:
    # Set class initialization
    def __init__(self, lr=1., decay=0., momentum=0.):
        # Store the assigned learning rate, current lr, decay rate, and iterations
        self.lr = lr
        self.lr_curr = lr
        self.decay = decay
        self.iteration = 0
        self.momentum = momentum

    # Call to update learning rate before any parameter update
    def preUpdateParams(self):
        # If there is a nonzero decay, update the lr before updating the parameters
        if self.decay:
            self.lr_curr = self.lr * (1. / (1. + self.decay * self.iteration))

    # Update our paramaters
    def updateParams(self, layer):
        # SGD with momentum calculation
        if self.momentum:
            # If layer does not have a momentum array, create it
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # No momentum array --> no bias array; so create it too.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum
            weight_updates = self.momentum * layer.weight_momentums - self.lr_curr * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates with momentum
            bias_updates = self.momentum * layer.bias_momentums - self.lr_curr * layer.dbiases
            layer.bias_momentums = bias_updates
        # SGD without momentum calculation
        else:
            weight_updates = -self.lr_curr * layer.dweights
            bias_updates = -self.lr_curr * layer.dbiases

        # With updates now calculated, update both weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call after a parameter update
    def postUpdateParams(self):
        self.iteration += 1

class AdaGrad(SGD):
    def __init__(self, lr=1., decay=0., epsilon=1e-7):
        super().__init__(lr, decay)
        self.epsilon = epsilon

    def updateParams(self, layer):
        # If layer does not contain cache arrays, create them
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with gradients squared
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # Vanilla SGD parameter update & norm
        layer.weights += -self.lr_curr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.lr_curr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

class RMSProp(SGD):
    def __init__(self, lr=1., decay=0., epsilon=1e-7, rho=0.9):
        super().__init__(lr, decay)
        self.epsilon = epsilon
        self.rho = rho

    def updateParams(self, layer):
        # Create cache arrays if not pre-existing
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        # Vanilla SGD parameter update & norm
        layer.weights += -self.lr_curr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.lr_curr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

class Adam(RMSProp):
    def __init__(self, lr=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        super().__init__(lr, decay, epsilon)
        self.beta1 = beta1
        self.beta2 = beta2

    def updateParams(self, layer):
        # Create cache arrays
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        # Correct momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iteration + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iteration + 1))
        # Update the cache with current gradients
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights ** 2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases ** 2

        # Correct cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iteration + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iteration + 1))

        # Vanilla SGD parameter update + norm with square rooted cache
        layer.weights += -self.lr_curr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.lr_curr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)


class DropoutLayer:
    # Method to initialize
    def __init__(self, rate):
        # Remember, we invert the rate
        self.rate = (1 - rate)

    # Forward pass method
    def forward(self, inputs):
        # Save the inputs
        self.inputs = inputs
        # Create mask and scale it
        self.binaryMask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply the mask to the outputs
        self.output = inputs * self.binaryMask

    # Backward pass method
    def backward(self, dvalues):
        # The gradient
        self.dinputs = dvalues * self.binaryMask


class Sigmoid:
    def forward(self, inputs):
        # Save input and calculate output
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # Calculate the derivative
        self.dinputs = dvalues * (1 - self.output) * self.output


class BinaryCrossEntropyLoss(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data on both sides to prevent division by 0, both sides to prevent skewing data
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise losses
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):
        # Sample size
        samples = len(dvalues)
        # Number of outputs per sample
        outputs = len(dvalues[0])

        # Clip data
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs /= samples