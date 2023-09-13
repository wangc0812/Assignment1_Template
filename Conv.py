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
        B, H1, W1, D1 = input.shape

        # Get weight dimensions
        k1, k2, _, D2 = self.weights.shape

        # Calculate output dimensions
        H2 = H1 - k1 + 1
        W2 = W1 - k2 + 1

        output = np.zeros((B, H2, W2, D2))

        # Perform convolution for each filter
        for n in range(B):
            for d2 in range(D2):
                for d1 in range(D1):
                    output[n, :, :, d2] += convolve2d(input[n, :, :, d1], self.weights[:, :, d1, d2], mode='valid')
                output[n, :, :, d2] += self.bias[d2]

                    # conv_result = convolve2d(input[n, :, :, d1], self.weights[:, :, d1, d2], mode='valid')
                    # print(conv_result.shape)
                    # print(output[n, d2, :, :].shape)
                    # output[n, d2, :, :] += self.bias[d2]
                    # print(self.weights.shape)  # 打印权重数组的形状
                    # print(d1, d2)  # 打印索引

        return output

    def backward(self, input_gradient):
        B, H1, W1, D2 = input_gradient.shape
        B, H, W, D1 = self.last_input.shape
        k1, k2, D1, D2 = self.weights.shape

        # calculate the gradient dL/dX

        # Initialize the output gradient
        output_gradient = np.zeros((B, H, W, D1))

        # Flip the weights for calculating gradient w.r.t. input
        flipped_weights = self.weights[::-1, ::-1, :, :]

        # Loop to calculate the gradient w.r.t. input
        for n in range(B):
            for d1 in range(D1):
                for d2 in range(D2):
                    output_gradient[n, :, :, d1] += convolve2d(input_gradient[n, :, :, d2],
                                                               flipped_weights[:, :, d1, d2], mode='full')

        # calculate the gradient dL/dW across miniBatch
            # Initialize weight gradients to zeros
            self.weight_gradient = np.zeros_like(self.weights)

            # Loop to calculate the gradient w.r.t weights
            for n in range(B):
                for d1 in range(D1):
                    for d2 in range(D2):
                        self.weight_gradient[:, :, d1, d2] += convolve2d(self.last_input[n, :, :, d1],
                                                                         input_gradient[n, :, :, d2], mode='valid')

        # calculate the gradient  dL/db across miniBatch
        self.bias_gradient = input_gradient.sum(axis=(0, 1, 2))

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
