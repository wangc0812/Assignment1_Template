from image_loader import load_mnist
from Conv import Conv
from FC import FC
from Pooling import AvgPool
from ReLU import ReLU
from Softmax import Softmax
from CrossEntropy import CrossEntropy
import numpy as np
import time

# load mnist dataset
train_data, train_label, test_data, test_label = load_mnist()
# x = [6000, 28, 28, 1]
print("Train data shape: {}, train laberl number: {}".format(train_data.shape, train_label.shape[0]))
print("Test data shape: {}, test laberl number: {}".format(test_data.shape, test_label.shape[0]))

# instantiate the network layers
conv1 = Conv(5, 5, 1, 8)
# kernel size: 5x5
# K1 = K2 = 5
# channel number = D1 = 1
# number of kernels = D2 = 8
relu1 = ReLU()
# ReLU activation
pool1 = AvgPool(2)
# window size 2 x 2
fc1 = FC(1152, 128)
# fully connected layer
# input size :1152
# output size :128
relu2 = ReLU()
fc2 = FC(128, 10)
softmax = Softmax()
loss_function = CrossEntropy()

# define the training params
batch_size = 256
N_iter = train_data.shape[0] // batch_size
lr = 0.02

print("Training start!")
start_time = time.time()
# train the network
for epoch in range(5):
    print("---------------------")
    for i in range(N_iter):
        input = train_data[i * batch_size:(i + 1) * batch_size, :, :, :]
        label = np.eye(10)[train_label[i * batch_size:(i + 1) * batch_size]]

        # forward propagation
        x = conv1.forward(input)  # we use multi-thread to accelerate training process
        x = relu1.forward(x)
        x = pool1.forward(x)
        x = x.reshape((batch_size, -1))  # Flatten the output for FC layer
        x = fc1.forward(x)
        x = relu2.forward(x)
        x = fc2.forward(x)
        x = softmax.forward(x)

        # Calculate loss
        L = loss_function.forward(x, label)

        # Print loss and accuracy for every epoch
        if epoch % 1 == 0 and i == 0:
            print("epoch: ", epoch)
            print("Loss: ", L)
            print("accuracy: ",
                  (batch_size - np.count_nonzero(np.argmax(label, axis=1) - np.argmax(x, axis=1))) / batch_size)

        # backpropagation
        grad = loss_function.backward()
        grad = softmax.backward(grad)
        grad = fc2.backward(grad)
        grad = relu2.backward(grad)
        grad = fc1.backward(grad)
        grad = grad.reshape((batch_size, 12, 12, 8))  # Reshape gradient to match pool1 output shape
        grad = pool1.backward(grad)
        grad = relu1.backward(grad)
        grad = conv1.backward(grad)

        # Weight update
        conv1.weight_update(lr)
        fc1.weight_update(lr)
        fc2.weight_update(lr)

end_time = time.time()
print("Training completed! Training Time:{}".format(end_time - start_time))
# test the network
N = 0
n = 0
N_iter = test_data.shape[0] // batch_size
print("Inference start")

for i in range(N_iter):
    input = test_data[i * batch_size:(i + 1) * batch_size, :, :, :]
    label = np.eye(10)[test_label[i * batch_size:(i + 1) * batch_size]]

    # inference
    x = conv1.forward_ori(input)  # we use single thread to reduce distribution cost
    x = relu1.forward(x)
    x = pool1.forward(x)
    x = x.reshape((batch_size, -1))  # Flattening the output for the fully connected layer
    x = fc1.forward(x)
    x = relu2.forward(x)
    x = fc2.forward(x)
    x = softmax.forward(x)

    # Get the label
    N += batch_size
    n += batch_size - np.count_nonzero(np.argmax(label, axis=1) - np.argmax(x, axis=1))

# calculate the accuracy
print("final accuracy for test is: ", n / N * 100, "%")
end_time2 = time.time()
print("Inference completed! Inference Time:{}".format(end_time2 - end_time))
