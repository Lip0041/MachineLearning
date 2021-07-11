import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def main():
    ex1()  # 不需要正则规划
    ex2()  # 需要正则规划


def ex1():
    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    # 处理数据
    data.insert(0, 'Ones', 1)
    X = np.matrix(data.iloc[:, 0: -1].values)
    y = np.matrix(data.iloc[:, -1:].values)
    theta = np.matrix(np.zeros(X.shape[1]))
    # 使用 scipy 中的 fmin_tnc 找最小代价对应的 theta
    res = optimize.fmin_tnc(cost, theta, gradient, args=(X, y, 0))
    print(cost(res[0], X, y))       # 计算代价：0.2034977015894744
    print_prediction(res, X, y)     # 利用这个最小theta, 计算预测准确率：accuracy = 89%
    # 绘制数据及决策边界：找出左上角和右下角的两点即可
    draw_data(data, 'Exam 1', 'Exam 2', 'Admitted', 'Not Admitted')
    x_point = [X[:, 1].min(), X[:, 1].max()]
    y_point = [-(X[:, 1].min() * res[0][1] + res[0][0]) / res[0][2],
               -(X[:, 1].max() * res[0][1] + res[0][0]) / res[0][2]]
    plt.plot(x_point, y_point, c='black', label='Boundary Line')
    plt.legend()
    plt.show()


def ex2():
    path = 'ex2data2.txt'   # 数据的边界曲线非线性
    data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

    X = np.matrix(data.iloc[:, 0: -1].values)
    y = np.matrix(data.iloc[:, -1:].values)
    XX = map_feature(X[:, 0].T, X[:, 1].T)  # 正则化：映射特征
    theta = np.matrix(np.zeros(XX.shape[1]))
    res = optimize.fmin_tnc(cost, theta, gradient, args=(XX, y, 0.1))   # 取正则化参数为0.1
    print(cost(res[0], XX, y))      # 最小代价：0.35238832740372816
    print_prediction(res, XX, y)    # accuracy = 99%
    # 绘制数据及决策边界：画出等高线
    draw_data(data, 'Test 1', 'Test 2', 'Accepted', 'Rejected')
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
    z = np.zeros((x1.size, x2.size))
    for i in range(0, x1.size):
        for j in range(0, x2.size):
            # 特征x1、x2 映射到正则化后的特征，并计算出每一组的代价
            z[i][j] = np.dot(map_feature(x1[i].reshape(1, -1), x2[j].reshape(1, -1)), res[0])
    z = np.transpose(z)
    plt.contour(x1, x2, z, [0], colors='black')
    plt.legend()
    plt.show()


def draw_data(data, x1, x2, y1, y2):
    # 提取出Admitted为1和0的数据
    # isin()接受一个列表，判断该列中元素是否在列表中，若在则将该行对应列设置为True
    positive = data[data[y1].isin([1])]
    negative = data[data[y1].isin([0])]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(positive[x1], positive[x2], s=50, c='b', marker='o', label=y1)
    ax.scatter(negative[x1], negative[x2], s=50, c='r', marker='x', label=y2)
    # 添加图例说明，即右上角的说明每种点种类的表
    ax.set_xlabel(x1 + ' Score')
    ax.set_ylabel(x2 + ' Score')


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


def map_feature(x1, x2):
    m = x1.shape[1]
    out = np.matrix(np.ones(m))
    degree = 6
    for i in range(1, degree + 1):
        for j in range(i + 1):
            add = np.multiply(np.power(x1, i - j), np.power(x2, j))
            out = np.vstack([out, add])
    return out.T


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]  # 根据定义，大于等于0.5即为1


def print_prediction(result, x, y):
    theta_min = np.matrix(result[0])
    predictions = predict(theta_min, x)     # 根据原数据以及计算得到的theta,预测应该的y值
    correct = [1 if (a == b) else 0 for (a, b) in zip(predictions, y)]  # 当预测值predictions与y值相等时，记入correct
    accuracy = (sum(map(int, correct)) % len(correct))  # 现在不懂map，不知道这是什么神仙操作
    print('accuracy = {0}%'.format(accuracy))


if __name__ == '__main__':
    main()
