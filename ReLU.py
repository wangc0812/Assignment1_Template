import numpy
import numpy as np
import scipy


class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        # Save the input for the backward calculation
        self.last_input = input

        # apply ReLU on input
        output = np.maximum(0, input)

        return output

    def backward(self, input_gradient):
        # Calculate gradient dL/dX
        output_gradient = input_gradient * (self.last_input > 0)

        return output_gradient
