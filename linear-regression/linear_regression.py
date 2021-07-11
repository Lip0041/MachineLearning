import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ex1()  # 单变量线性回归
    ex2()  # 多变量线性回归


def ex1():
    path = 'ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    X, y = process_data(data)  # 提取其中的数据为matrix形式
    theta = np.matrix(np.array([0, 0]))  # 初始化 theta
    g, cost = gradient_decent(X, y, theta, alpha=0.01, iteration=1000)  # 梯度下降
    draw_model1(data, g)  # 画出线性拟合
    draw_3d(X, y)  # 绘制theta与J的关系
    print(normal_eqn(X, y))  # 正规方程求解


def ex2():
    path = 'ex1data2.txt'
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    data = (data - data.mean()) / data.std()  # 归一化特征，其中 mean()获取平均值，std()获取标准差
    # 以下同上
    X, y = process_data(data)
    theta = np.matrix(np.array([0, 0, 0]))
    g, cost = gradient_decent(X, y, theta, alpha=0.01, iteration=1000)
    draw_model2(data, g)
    print(normal_eqn(X, y))


def process_data(data):
    data.insert(0, 'Ones', 1)   # 在第一列之前添加一列1
    X = data.iloc[:, 0: -1]
    y = data.iloc[:, -1:]
    # 将X, y转换为矩阵
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    return X, y


def compute_cost(X, y, theta):
    # 计算cost
    return np.mean(np.power(((X * theta.T) - y), 2)) / 2


def gradient_decent(X, y, theta, alpha, iteration):
    # 梯度下降
    cost = np.zeros(iteration)
    # 每一次迭代，更新theta
    for i in range(iteration):
        error = (X * theta.T) - y
        temp = theta.T - (X.T * error) / len(X) * alpha
        theta = temp.T
        cost[i] = compute_cost(X, y, theta)
    # 画出代价函数与迭代次数的关系图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iteration), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()
    return theta, cost


def draw_model1(data, g):
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)
    # 画出线性拟合
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    # 画出原数据
    ax.scatter(data.Population, data.Profit, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()


def draw_model2(data, g):
    x1 = np.linspace(data.Size.min(), data.Size.max())
    x2 = np.linspace(data.Bedrooms.min(), data.Bedrooms.max())
    x1, x2 = np.meshgrid(x1, x2)
    h = g[0, 0] + g[0, 1] * x1 + g[0, 2] * x2
    # 画出线性拟合
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x1, x2, h, cmap='rainbow')
    # 画出原数据
    a = data.iloc[:, 1: 2]
    b = data.iloc[:, 2: 3]
    c = data.iloc[:, 3: 4]
    ax.scatter(a, b, c, c='r')
    ax.set_xlabel('Size')
    ax.set_ylabel('Bedrooms')
    ax.set_zlabel('Price')
    ax.set_title('Price vs. Size and Bedrooms')
    plt.show()


def draw_3d(X, y):
    # 参考链接：
    # https://www.pythonheidong.com/blog/article/576895/340066ed356e3e68be33/
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
    # contour，等高线绘制
    plt.figure()
    contour = np.logspace(-3, 3, 20)    # 以10**-3次方（0.001）和10**3（1000）之间的对数间隔绘制20条等高线
    z = plt.contour(theta0, theta1, J, levels=contour)
    plt.clabel(z)   # 显示每条等高线的代价函数值
    plt.show()


def normal_eqn(X, y):
    # 正规方程求解 theta = (X^TX)^(-1)X^Ty
    # tip: X@T <--> X.dot(T)
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)   # theta is n*1
    return theta.T


if __name__ == '__main__':
    main()
