import numpy as np
from matplotlib import pyplot as plt

def create_data(samples, classes):
    N = samples # 100 - number of points per class
    D = 2  # dimensionality
    K = classes # 3  # number of classes - blue, red and yellow
    X = np.zeros((N*K, D))
    y = np.zeros(N*K, dtype='uint8') # class labels
    log("X: %s", X)
    log("y: %s", y)

    for j in range(K):  # for each class
        log("j: %s", j)
        index = range(N*j, N*(j+1))
        log("ix: %s", index)
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) * 0.2  # theta
        log("r: %s", r)
        log("t: %s", t)
        X[index] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[index] = j

    # visualize it
    #plt.plot('scatter', X[:, 0], X[:, 0], c=y, s=40, cmap=plt.cm.Spectral)
    #plt.show()
    return X, y

def log(prefix, value):
    s = prefix + str(value)
    #print(prefix % value)

#create_data(samples=100, classes=3)
