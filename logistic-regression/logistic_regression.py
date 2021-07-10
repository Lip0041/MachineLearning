import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def draw_data(data, x1, x2, y1, y2):
    # 提取出Admitted为1和0的数据
    # isin()接受一个列表，判断该列中元素是否在列表中，若在则将该行对应列设置为True
    positive = data[data[y1].isin([1])]
    negative = data[data[y1].isin([0])]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive[x1], positive[x2], s=50, c='b', marker='o', label=y1)
    ax.scatter(negative[x1], negative[x2], s=50, c='r', marker='x', label=y2)
    # 添加图例说明，即右上角的说明每种点种类的表
    ax.legend()
    ax.set_xlabel(x1 + ' Score')
    ax.set_ylabel(x2 + ' Score')
    plt.show()


def ex_data1():
    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
    draw_data(data, 'Exam1', 'Exam2', 'Admitted', 'Not Admitted')
    data.insert(0, 'Ones', 1)
    data.iloc[:, 1: -1] = (data.iloc[:, 1: -1] - data.iloc[:, 1: -1].mean()) / data.iloc[:, 1: -1].std()
    X = data.iloc[:, 0: -1]
    y = data.iloc[:, -1:]
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.zeros(X.shape[1]))
    # gradient descend
    res = gradient_descent(theta, X, y)
    print(compute_cost(res, X, y))
    # 利用这个最小theta, 计算预测准确率
    print_prediction(res, X, y)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compute_cost(theta, X, y):
    theta = np.matrix(theta)
    h = sigmoid(X * theta.T)
    return np.mean(np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h)))


def gradient_descent(theta, X, y, learningRate=1, iteration=100):
    cost = np.zeros(iteration)
    for i in range(iteration):
        error = sigmoid(X * theta.T) - y
        temp = theta.T - (X.T * error) / len(X) * learningRate
        theta = temp.T
        cost[i] = compute_cost(theta, X, y)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(np.arange(iteration), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()
    return theta


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def cost_regularized(theta, X, y, lambda0):
    theta = np.matrix(theta)
    h = sigmoid(X * theta.T)
    reg = lambda0 / 2.0 / len(X) * np.mean(np.multiply(theta, theta))
    return np.mean(np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))) + reg


def gradient(theta, X, y, lambda0):
    error = sigmoid(X * theta.T) - y
    grad = (X.T * error) / len(X)
    for i in range(theta.shape[1]):
        if i == 0:
            continue
        else:
            grad[i, 0] += theta[0, i] * lambda0 / len(X)
    return grad.T


def ex_data2():
    path = 'ex2data2.txt'
    data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
    # draw_data(data, 'Test 1', 'Test 2', 'Accepted', 'Rejected')

    data.insert(0, 'Ones', 1)
    # data.iloc[:, 1: -1] = (data.iloc[:, 1: -1] - data.iloc[:, 1: -1].mean()) / data.iloc[:, 1: -1].std()
    X = data.iloc[:, 0: -1]
    y = data.iloc[:, -1:]
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.zeros(X.shape[1]))
    print(cost_regularized(theta, X, y, 0.1))
    res = gradient(theta, X, y, 0.1)
    # print prediction
    result = optimize.fmin_bfgs(cost_regularized, theta, fprime=gradient, args=(X, y, 0.1))
    print_prediction(result, X, y)

    # degree = 5
    # x1 = data['Test 1']
    # x2 = data['Test 2']
    #
    # data.insert(3, 'Ones', 1)
    #
    # for i in range(1, degree):
    #     for j in range(0, i):
    #         data['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)
    #
    # data.drop('Test 1', axis=1, inplace=True)
    # data.drop('Test 2', axis=1, inplace=True)
    #
    # cols = data.shape[1]
    # x = np.array(data.iloc[:, 1: cols].values)
    # y = np.array(data.iloc[:, 0: 1].values)
    # theta = np.matrix(np.zeros(11))
    # print(compute_cost(theta, x, y))
    # res = gradient(theta, x, y, 0.1)
    # # print prediction
    # print_prediction(res, x, y)


def print_prediction(result, x, y):
    theta_min = np.matrix(result[0])
    predictions = predict(theta_min, x)
    correct = [1 if ((a == 1) and (b == 1) or (a == 0) and (b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))


if __name__ == '__main__':
    # ex_data1()
    ex_data2()
