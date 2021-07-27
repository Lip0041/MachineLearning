import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.io import loadmat


def main():
    X, y = load_data('ex3data1.mat')
    # plot_image(X)
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
    t = logistic_regression(X, y, lambda0=1)
    print(t.shape)
    y_pred = predict(X, t)
    print('Accuracy={}'.format(np.mean(y[0] == y_pred)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cost(theta, X, y, lambda0=0):
    # theta: 1 * n, X: m * n, y: m * 1
    theta = np.matrix(theta)
    h = sigmoid(X * theta.T)
    reg = lambda0 / 2.0 / len(X) * np.mean(np.multiply(theta, theta))
    return -np.mean(np.multiply(y, np.log(h)) + np.multiply((1 - y), np.log(1 - h))) + reg


def gradient(theta, X, y, lambda0=0):
    # theta: 1 * n, X: m * n, y: m * 1
    theta = np.matrix(theta)
    error = sigmoid(X * theta.T) - y
    grad = (X.T * error).T / len(X)
    for i in range(theta.shape[1]):
        if i == 0:
            continue
        else:
            grad[0, i] += theta[0, i] * lambda0 / len(X)
    return grad


def logistic_regression(X, y, lambda0=1):
    theta = np.matrix(np.zeros(X.shape[1]))
    X = np.matrix(X)
    y = np.matrix(y)
    res = optimize.fmin_tnc(cost, theta, gradient, args=(X, y, lambda0))
    res = res[0]
    return res


def predict(x, theta):
    prob = sigmoid(x.T * theta)
    return (prob >= 0.5).astype(int)


def load_data(path, transpose=True):
    data = loadmat(path)
    y = data.get('y')
    y = y.reshape(y.shape[0])

    X = data.get('X')
    if transpose:
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])
    return X, y


def plot_image(X):
    size = int(np.sqrt(X.shape[1]))

    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
    sample_images = X[sample_idx, :]

    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8,8))
    for r in range(10):
        for c in range(10):
            ax[r, c].matshow(sample_images[10 * r + c].reshape(size, size), cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()


if __name__ == '__main__':
    main()