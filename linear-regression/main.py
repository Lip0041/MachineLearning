import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradient_decent(X, y, theta, alpha, iters):
    # print(theta.shape)
    temp = np.matrix(np.zeros(theta.shape))
    # print(temp)
    parameters = int(theta.ravel().shape[1])
    # print(theta.ravel())
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = compute_cost(X, y, theta)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()
    return theta, cost


def exp1_1():
    path = 'ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    # print(data.head())
    # print(data.describe())
    # data.info()
    # data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
    # plt.show()

    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]
    # print(data.head())
    # print(X.head())
    # print(y.head())
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))
    alpha = 0.01
    iters = 1000
    # print(theta)
    g, cost = gradient_decent(X, y, theta, alpha, iters)
    # print(g)
    # print(compute_cost(X, y, g))
    draw_linear_model(data, g)
    # normal_theta = normal_eqn(X, y)
    # draw_linear_model(data, normal_theta)
    # print(cost)
    # print(g)
    draw_3d(X, y)


def draw_3d(X, y):
    J = np.zeros((100, 100))
    theta0 = np.linspace(-10, 10, 100)
    theta1 = np.linspace(-1, 4, 100)
    for i in range(len(theta0)):
        for j in range(len(theta1)):
            z = np.matrix((theta0[i], theta1[j]))
            J[i, j] = compute_cost(X, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(theta0, theta1, J, cmap="rainbow")
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('J')
    plt.show()
    # contour
    plt.figure()
    contour = np.logspace(-2, 3, 20)
    z = plt.contour(theta0, theta1, J, levels=contour)
    plt.clabel(z)
    plt.show()


def exp1_2():
    path = 'ex1data2.txt'
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    # 归一化特征
    data = (data - data.mean()) / data.std()
    # print(data.mean())
    # print(data.std())
    data.insert(0, 'Ones', 1)

    cols = data.shape[1]
    X = data.iloc[:, 0: cols-1]
    y = data.iloc[:, cols-1: cols]

    # convert to matrices
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0, 0]))
    g, cost = gradient_decent(X, y, theta, alpha=0.01, iters=1000)
    # print(data)
    draw_model(data, g, cost)
    # print(cost.shape)


def draw_linear_model(data, g):
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()


def draw_model(data, g, cost):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x1 = np.linspace(data.Size.min(), data.Size.max())
    x2 = np.linspace(data.Bedrooms.min(), data.Bedrooms.max())
    x1, x2 = np.meshgrid(x1, x2)
    h = g[0, 0] + g[0, 1] * x1 + g[0, 2] * x2
    # print(h.shape)
    # print(data.Price.values)
    ax.plot_surface(x1, x2, h, cmap='rainbow')
    a = data.iloc[:, 1: 2]
    b = data.iloc[:, 2: 3]
    c = data.iloc[:, 3: 4]
    ax.scatter(a, b, c, c='r')

    print(h.shape)
    # print(a)
    plt.show()


def normal_eqn(X, y):
    # X@T <--> X.dit(T)
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta.T


def main():
    exp1_1()
    # exp1_2()


if __name__ == '__main__':
    main()
