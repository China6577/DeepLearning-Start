import numpy as np


class AffineLayer:
    """
    仿射层（全连接层）：
        计算加权信号的总和
    """

    def __init__(self, W, b):
        self.W, = W
        self.b = b
        self.x = None
        self.dW = None  # self.dW 和 self.db 可以用于后续的参数更新
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        """
        计算梯度矩阵dout更新后的梯度矩阵dx
        :param dout: 从后续层传来的梯度，形状通常为 (batch_size, output_size)
        :return: 关于输入x的梯度矩阵
        """
        dx = np.dot(dout, self.W.T)  # 计算关于输入x的梯度
        self.dW = np.dot(self.x.T, dout)  # 计算关于权重W的梯度
        self.db = np.sum(dout, axis=0)  # 计算关于偏置b的梯度 axis=0是沿着行对列求和

        return dx
