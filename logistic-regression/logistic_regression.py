import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt    # 使用scipy中的truncated newton(TNC)实现寻找最优参数


def draw_data(data, x1, x2, y1, y2):
    # 提取出Admitted为1和0的数据
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
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

    draw_data(data, 'Exam 1', 'Exam 2', 'Admitted', 'Not Admitted')

    data.insert(0, 'Ones', 1)
    X = data.iloc[:, 0: data.shape[1] - 1]
    y = data.iloc[:, data.shape[1] - 1: data.shape[1]]
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(3)
    # theta is 0, the cost is:
    cost(theta, X, y)
    # gradient descend
    gradient(theta, X, y)
    # 使用TNC寻找最优参数
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
    # 利用这个最小theta, 计算预测准确率
    print_prediction(result, X, y)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, learningRate=0):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1: theta.shape[1], 2]))
    return np.sum(first - second) / len(X) + reg


def gradient(theta, X, y, learningRate=0):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    # 获取参数的个数
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if i == 0:
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X) + (learningRate / len(X)) * theta[:, i])
    return grad


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def ex_data2():
    path = 'ex2data2.txt'
    data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

    draw_data(data, 'Test 1', 'Test 2', 'Accepted', 'Rejected')
    degree = 5
    x1 = data['Test 1']
    x2 = data['Test 2']

    data.insert(3, 'Ones', 1)

    for i in range(1, degree):
        for j in range(0, i):
            data['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

    data.drop('Test 1', axis=1, inplace=True)
    data.drop('Test 2', axis=1, inplace=True)

    cols = data.shape[1]
    x = np.array(data.iloc[:, 1: cols].values)
    y = np.array(data.iloc[:, 0: 1].values)
    theta = np.zeros(11)
    cost(theta, x, y, learningRate=1)
    gradient(theta, x, y, learningRate=1)
    # TNC find optimum
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y, 1))
    # print prediction
    print_prediction(result, x, y)


def print_prediction(result, x, y):
    theta_min = np.matrix(result[0])
    predictions = predict(theta_min, x)
    correct = [1 if ((a == 1) and (b == 1) or (a == 0) and (b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))


if __name__ == '__main__':
    # ex_data1()
    ex_data2()
