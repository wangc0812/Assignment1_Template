import numpy
import numpy as np
import scipy


class FC():
    def __init__(self, D1, D2):
        self.weights = np.random.randn(D1, D2) * np.sqrt(2 / (D1))
        self.bias = np.random.randn(D2)
        self.last_input = None
        self.ok_to_update = False
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, input):
        # save the input for gradient calculation
        self.last_input = input

        # fc layer forward calculation
        # output = self.weights.T @ input + self.bias
        output = np.dot(input, self.weights) + self.bias
        # weight size D1 X D2

        self.ok_to_update = True

        return output

    def backward(self, input_gradient):
        # calculate the gradient dL/dX
        output_gradient = np.dot(input_gradient, self.weights.T)

        # calculate the gradient dL/dW across miniBatch
        self.weight_gradient = np.dot(self.last_input.T, input_gradient)

        # calculate the gradient dL/db across miniBatch
        self.bias_gradient = np.sum(input_gradient, axis=0)

        self.ok_to_update = True
        return output_gradient

    def weight_update(self, lr=0.1):
        if self.ok_to_update:
            # update the weights and bias
            self.weights -= lr * self.weight_gradient
            self.bias -= lr * self.bias_gradient

            self.ok_to_update = False
