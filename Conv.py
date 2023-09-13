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
        for n in range(B):  # each batch
            for d2 in range(D2):  # each output channel / kernel
                for d1 in range(D1): # each input channel
                    output[n, :, :, d2] += convolve2d(input[n, :, :, d1], self.weights[:, :, d1, d2], mode='valid')
                output[n, :, :, d2] += self.bias[d2]

        # conv_result = convolve2d(input[n, :, :, d1], self.weights[:, :, d1, d2], mode='valid')
        # print(conv_result.shape)
        # print(output[n, d2, :, :].shape)
        # output[n, d2, :, :] += self.bias[d2]
        # print(self.weights.shape)  # weight size
        # print(d1, d2)  # index

        return output

    def backward(self, input_gradient):
        B, H1, W1, D2 = input_gradient.shape  # 获取输入梯度的形状
        B, H, W, D1 = self.last_input.shape  # 获取上一次输入的形状
        k1, k2, D1, D2 = self.weights.shape  # 获取权重的形状

        output_gradient = np.zeros_like(self.last_input)

        # Padding input_gradient for convolution
        input_gradient_padded = np.pad(input_gradient, ((0, 0), (k1 - 1, k1 - 1), (k2 - 1, k2 - 1), (0, 0)),
                                       mode='constant')

        # Initialize gradients
        self.weight_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)

        # calculate the gradients
        for n in range(B):
            for d2 in range(D2):
                self.bias_gradient[d2] += np.sum(input_gradient[n, :, :, d2])

                for d1 in range(D1):
                    self.weight_gradient[:, :, d1, d2] += convolve2d(self.last_input[n, :, :, d1],
                                                                     input_gradient[n, :, :, d2], mode='valid')
                    output_gradient[n, :, :, d1] += convolve2d(input_gradient_padded[n, :, :, d2],
                                                               np.rot90(self.weights[:, :, d1, d2], 2), mode='valid')

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
