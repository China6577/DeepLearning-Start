import numpy as np


# 隐藏层激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 生成和x一样大小的矩阵


# 输出层激活函数
def softmax(x):
    y = np.max(x)  # 防止指数溢出
    exp_x = np.exp(x - y)
    exp_x_sum = np.sum(np.exp(x))
    return exp_x / exp_x_sum


def cross_entropy_error(y, t):
    """
    交叉熵误差
    :param y: 预测数据
    :param t: 测试数据
    :return:输出的值越小越好
    """
    delta = 1e-7  # 防止log(0)变为负无穷大 1e-7=0.0000001
    return -np.sum(t * np.log(y + delta))


# 数值法计算梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # 生成和x形状相同的数组 值全为0

    for i in range(x.size):
        tmp_val = x[i]

        x[i] = tmp_val + h
        fxh1 = f(x)
        x[i] = tmp_val - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2 * h)

        x[i] = tmp_val

    return grad
