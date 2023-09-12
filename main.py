from image_loader import load_mnist
from Conv import Conv
from FC import FC
from Pooling import AvgPool
from ReLU import ReLU
from Softmax import Softmax
from CrossEntropy import CrossEntropy
import numpy as np

# load mnist dataset
train_data, train_label, test_data, test_label = load_mnist()

print("Train data shape: {}, train laberl number: {}".format(train_data.shape, train_label.shape[0]))
print("Test data shape: {}, test laberl number: {}".format(test_data.shape, test_label.shape[0]))

# instantiate the network layers
conv1 = Conv(k1=5, k2=5, D1=1, D2=32)
relu1 = ReLU()
pool1 = AvgPool(k=2)

conv2 = Conv(k1=5, k2=5, D1=32, D2=64)
relu2 = ReLU()
pool2 = AvgPool(k=2)

fc1 = FC(D1=7 * 7 * 64, D2=128)
relu3 = ReLU()

fc2 = FC(D1=128, D2=10)
softmax = Softmax()
cross_entropy = CrossEntropy()



# define the training params
batch_size = 256
N_iter = train_data.shape[0] // batch_size
lr = 0.02

# train the network
for epoch in range(5):
    for i in range(N_iter):
        input = train_data[i * batch_size:(i + 1) * batch_size, :, :, :]
        label = np.eye(10)[train_label[i * batch_size:(i + 1) * batch_size]]

        # forwardpropagation
        conv1_output = conv1.forward(input)
        relu1_output = relu1.forward(conv1_output)
        pool1_output = pool1.forward(relu1_output)

        conv2_output = conv2.forward(pool1_output)
        relu2_output = relu2.forward(conv2_output)
        pool2_output = pool2.forward(relu2_output)

        fc_input = pool2_output.reshape((batch_size, -1))

        fc1_output = fc1.forward(fc_input)
        relu3_output = relu3.forward(fc1_output)

        fc2_output = fc2.forward(relu3_output)
        x = softmax.forward(fc2_output)

        cross_entropy_loss = CrossEntropy.forward(x, label)
        L = np.mean(cross_entropy_loss)

        if epoch % 1 == 0 and i == 0:
            print("epoch: ", epoch)
            print("Loss:", L)
            accuracy = (batch_size - np.count_nonzero(
                np.argmax(label, axis=1) - np.argmax(x, axis=1))) / batch_size
            print("Accuracy:", accuracy)
            # print(L)
            # print((batch_size - np.count_nonzero(np.argmax(label, axis=1) - np.argmax(x, axis=1))) / batch_size)

        # backpropagation
        cross_entropy_gradient = CrossEntropy.backward(label)  # Gradient of loss w.r.t. softmax output

        # Backpropagate through fully connected layers
        fc2_gradient = fc2.backward(cross_entropy_gradient)
        relu3_gradient = relu3.backward(fc2_gradient)
        fc1_gradient = fc1.backward(relu3_gradient)

        # Reshape gradient for convolution layers
        pool2_gradient = fc1_gradient.reshape(pool2_output.shape)

        # Backpropagate through convolution and pooling layers
        conv2_gradient = pool2.backward(pool2_gradient)
        relu2_gradient = conv2.backward(conv2_gradient)

        pool1_gradient = relu2_gradient
        conv1_gradient = pool1.backward(pool1_gradient)
        relu1_gradient = conv1.backward(conv1_gradient)

        # weight update
        conv1.weight_update(lr)
        conv2.weight_update(lr)
        fc1.weight_update(lr)
        fc2.weight_update(lr)

# test the network
N = 0
n = 0
N_iter = test_data.shape[0] // batch_size

for i in range(N_iter):
    input = test_data[i * batch_size:(i + 1) * batch_size, :, :, :]
    label = np.eye(10)[test_label[i * batch_size:(i + 1) * batch_size]]

    # inference
    conv1_output = conv1.forward(input)
    relu1_output = relu1.forward(conv1_output)
    pool1_output = pool1.forward(relu1_output)

    conv2_output = conv2.forward(pool1_output)
    relu2_output = relu2.forward(conv2_output)
    pool2_output = pool2.forward(relu2_output)

    fc_input = pool2_output.reshape((batch_size, -1))

    fc1_output = fc1.forward(fc_input)
    relu3_output = relu3.forward(fc1_output)

    fc2_output = fc2.forward(relu3_output)
    x = softmax.forward(fc2_output)

    # get the label
    N += batch_size
    n += batch_size - np.count_nonzero(np.argmax(label, axis=1) - np.argmax(x, axis=1))

# calculate the accuracy
print("final accuracy for test is: ", n / N * 100, "%")
