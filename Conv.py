import numpy
import numpy as np
from scipy.signal import convolve2d


class Conv():
    def __init__(self, k1, k2, D1, D2):
        self.weights = np.random.randn(k1, k2, D1, D2) * np.sqrt(2 / (k1 * k2 * D1))
        self.bias = np.random.randn(D2)
        self.last_input = None
        self.ok_to_update = False
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, input):
        # save the input for gradient calculation
        self.last_input = input

        # Get input shape
        B, H, W, D1 = input.shape

        # Get filter dimensions
        k1, k2, _, D2 = self.weights.shape

        # Calculate output dimensions
        output_height = H - k1 + 1
        output_width = W - k2 + 1

        # Initialize output
        output = np.zeros((B, output_height, output_width, D2))

        # Perform convolution for each filter
        for i in range(D2):  # for each output channel
            for j in range(D1):  # for each input channel
                output[:, :, :, i] += convolve2d(input[:, :, :, j], self.weights[:, :, j, i], mode='valid')
        # Add bias
        output += self.bias[None, None, None, :]

        self.ok_to_update = True

        return output

    def backward(self, input_gradient):
        B, H1, W1, D2 = input_gradient.shape
        B, H, W, D1 = self.last_input.shape
        k1, k2, D1, D2 = self.weights.shape

        # Initialize gradients
        self.weight_gradient = np.zeros((k1, k2, D1, D2))

        # calculate the gradient dL/dX
        output_gradient = np.zeros((B, H, W, D1))
        for i in range(D2):
            for j in range(D1):
                for b in range(B):
                    output_gradient[b, :, :, j] += np.correlate(input_gradient[b, :, :, i], self.weights[:, :, j, i],
                                                                mode='full')

        # calculate the gradient dL/dW across miniBatch
        for i in range(D2):
            for j in range(D1):
                self.weight_gradient[:, :, j, i] = np.sum(
                    [np.correlate(self.last_input[b, :, :, j], input_gradient[b, :, :, i], mode='valid')
                     for b in range(B)], axis=0
                ) / B

        # calculate the gradient  dL/db across miniBatch
        self.bias_gradient = np.sum(input_gradient, axis=(0, 1, 2)) / B

        self.ok_to_update = True
        return output_gradient

    def weight_update(self, lr=0.1):
        if self.ok_to_update:
            # update the weights and bias
            # update the weights using the weight gradients
            self.weights -= lr * self.weight_gradient

            # update the bias using the bias gradients
            self.bias -= lr * self.bias_gradient

            # reset the ok_to_update flag
            self.ok_to_update = False
        else:
            print("No gradients available for update. ")
