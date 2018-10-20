import os
import struct
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
from scipy.spatial import distance

def read(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return (lbl, img)
	
def soft_max_function(W,X):
    numerator = np.exp(np.dot(W.T, X))
    denominator = np.sum(numerator, axis = 0)
    return (numerator/denominator).T
	
def predict_accuracy(W, X, Y):
    accuracy_counter = 0
    predicted_values = soft_max_function(W, X)
    indices = np.argmax(predicted_values, axis=1)
    for i in range(len(indices)):
        if Y[i] == indices[i]:
            accuracy_counter += 1
    return accuracy_counter
	
def transform_target(label):
    zero_matrix = np.zeros((label.shape[0], 10))
    zero_matrix[np.arange(label.shape[0]), label] = 1
    return zero_matrix
	
y_train, x_train = read("training", "./MNIST")
y_test, x_test = read("testing", "./MNIST")
x_train = x_train.astype(np.int64)
x_test = x_test.astype(np.int64)

x_train = [train_image.reshape(784) for train_image in x_train]
x_train = np.array(x_train)

x_test = [test_image.reshape(784) for test_image in x_test]
x_test = np.array(x_test)
print(x_train.shape)
print(x_test.shape)

#initalizing parameter W with all zeros
learning_rate = 1e-4
accuracy_cache = [0] * 100
W = np.zeros((784, 10))
x_train = x_train.T
x_test = x_test.T
y_train = transform_target(y_train)
for i in range(100):
    #getting the predictions using soft max function
    predictions = soft_max_function(W, x_train)
    gradient = (np.dot(x_train, np.subtract(y_train, predictions))) / x_train.shape[1]
    W[:, :9] = W[:, :9] + (learning_rate * gradient)[:, :9]
    accuracy_cache[i] = predict_accuracy(W, x_test, y_test)
accuracy_cache = [x/10000 for x in accuracy_cache]
print(accuracy_cache)
pyplot.plot([x for x in range(1,101)], accuracy_cache)
pyplot.show()