import numpy
import numpy as np
import scipy

class CrossEntropy():
    def __init__(self):
        pass

    def forward(self,input, label):
        #avoid zero input
        input = input + 1e-8

        #save input/label for gradient calculation
        self.last_input = input
        self.last_label = label

        #calculate crossentropy between input and label
        output = -np.sum(label * np.log(input) + (1 - label) * np.log(1 - input)) / input.shape[0]

        return output

    def backward(self):
        #calculate gradient dL/dX
        output_gradient = (-self.last_label / (self.last_input + 1e-8)) + (1 - self.last_label) / (
                    1 - self.last_input + 1e-8)

        # Normalize gradient across the batch
        output_gradient /= self.last_input.shape[0]

        return output_gradient


