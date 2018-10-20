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
	

k_nn = [1,3,5,10,30,50,70,80,90,100]
y_train, x_train = read("training", "./MNIST")
y_test, x_test = read("testing", "./MNIST")
x_train = x_train.astype(np.int64)
x_test = x_test.astype(np.int64)

x_train = [train_image.reshape(784) for train_image in x_train]
x_train = np.array(x_train)

x_test = [test_image.reshape(784) for test_image in x_test]
x_test = np.array(x_test)

euclidean_distance = distance.cdist(x_test, x_train)
sorted_distances = np.argsort(euclidean_distance, axis=1)

accuracy_cache = [0] * 10
for k_index in range(len(k_nn)):
    indexes_neigh = sorted_distances[:, :k_nn[k_index]]
    for i in range(len(indexes_neigh)):
        label_list = []
        for index in indexes_neigh[i]:
            label_list.append(y_train[index])
        if y_test[i] == max(label_list,key=label_list.count):
            accuracy_cache[k_index] = accuracy_cache[k_index] + 1

accuracy_ratio = [value/10000 for value in accuracy_cache]
print(accuracy_ratio)
pyplot.plot(k_nn, accuracy_ratio)
pyplot.show()