import numpy as np


class SoftmaxWithLossLayer:
    """
    正规化和损失层
    """
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None  # 监督数据 以one-hot形式

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # 反向传播要除以批的大小 传递给前面的层单个数据的误差

        return dx


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
