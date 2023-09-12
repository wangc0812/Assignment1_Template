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

        output = np.zeros((B, OH, OW, D1))

        # Apply average pool of input
        for b in range(B):
            for i in range(0, H, self.k):
                for j in range(0, W, self.k):
                    for d in range(D1):
                        output[b, i // self.k, j // self.k, d] = np.mean(input[b, i:i + self.k, j:j + self.k, d])

        return output

    def backward(self, input_gradient):
        # Dimensions of the input array
        B, H, W, D1 = input_gradient.shape

        # Dimensions of the output array
        OH = H * self.k
        OW = W * self.k

        output_gradient = np.zeros((B, OH, OW, D1))

        # calculate gradient dL/dX
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    for d in range(D1):
                        output_gradient[b, i * self.k:(i + 1) * self.k, j * self.k:(j + 1) * self.k, d] = \
                        input_gradient[b, i, j, d] / (self.k * self.k)

        return output_gradient
