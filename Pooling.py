import numpy
import numpy as np
import scipy


class AvgPool():
    def __init__(self, k):
        self.k = k

    def forward(self, input):
        # Dimensions of the input array
        B, H, W, D1 = input.shape

        # Dimensions of the output array
        OH = H // self.k
        OW = W // self.k

        if H % self.k != 0 or W % self.k != 0:
            raise ValueError("Input dimensions must be divisible by the pooling size")

        output = input.reshape(B, OH, self.k, OW, self.k, D1).mean(axis=(2, 4))

        return output

    def backward(self, input_gradient):
        # Dimensions of the input array
        B, H, W, D1 = input_gradient.shape

        # Dimensions of the output array
        OH = H * self.k
        OW = W * self.k

        output_gradient = np.zeros((B, OH, OW, D1))

        # calculate gradient dL/dX
        output_gradient = np.repeat(np.repeat(input_gradient, self.k, axis=1), self.k, axis=2) / (self.k ** 2)

        return output_gradient
