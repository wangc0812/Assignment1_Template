import numpy
import numpy as np
import scipy


class Softmax():
    def __init__(self):
        pass

    def forward(self, input):
        # calculate the softmax of input
        exp_input = np.exp(input - np.max(input, axis=-1, keepdims=True))
        output = exp_input / np.sum(exp_input, axis=-1, keepdims=True)

        # save the cross entropy for gradient calculation
        self.last_output = output

        return output

    def backward(self, input_gradient):
        # calculate gradient dL/dX
        output_gradient = self.last_output * (
                input_gradient - np.sum(self.last_output * input_gradient, axis=-1, keepdims=True))

        return output_gradient
