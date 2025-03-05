from CNN深度学习架构.神经网络层.functions import *
import numpy as np


class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self, dout):
        # sigmoid函数的反向传播的梯度公式为 dy = 误差函数L对y的求导*y*(y-1)
        dx = dout * (1 - self.out) * self.out

        return dx


class ReluLayer:
    """
    relu层(以relu作为激活函数)：
        1.当x小于等于0则输出0
        2.x大于0则输出x
    """

    def __init__(self):
        self.mask = None  # mask为实例变量 由forward函数和backward函数共享

    def forward(self, x):
        """
        relu函数的实现
        :param x: 输入的矩阵
        :return: 输入矩阵通过relu正向传播后的矩阵
        """
        self.mask = (x <= 0)  # mask由布尔值构成的numpy数组（矩阵） 当x小于等于0是为ture
        out = x.copy()
        out[self.mask] = 0  # 为ture则设置为0 false则不变

        return out

    def backward(self, dout):
        """
        反向传播法求梯度
        :param dout: 从后续层传来的梯度矩阵
        :return: 梯度矩阵通过relu反向传播后的梯度矩阵
        """
        dout[self.mask] = 0  # 如果x小于等于0则梯度值为0
        dx = dout  # x等于等于0则梯度值不变

        return dx


class AffineLayer:
    """
    仿射层（全连接层）：
        计算加权信号的总和
    """

    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        self.dW = None  # self.dW 和 self.db 可以用于后续的参数更新
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
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

        dx = dx.reshape(self.original_x_shape)  # 还原x的原始形状
        return dx


class SoftmaxWithLossLayer:
    """
    正规化和损失层
    """

    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出结果
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size  # 梯度=预测数据-测试数据 反向传播要除以批的大小 传递给前面的层单个数据的误差
        else:  # 监督数据是存储正确索引的一维数组
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1  # 生成一个从0到batch_size-1的整数数组 并将该数组和t数组对应 如[0,1,2] [2,1,3]
                                                    # 变为[0,2] [1,1] [2,3]这些是预测结果矩阵正确结果预测的概率的位置
                                                    # 在交叉熵损失函数的梯度计算中 对于每个样本的真实类别对应的预测概率
                                                    # 需要减去1(用梯度=预测数据-测试数据测试数据的概率为1 因为是真实的类别在t中)来得到梯度
            dx = dx / batch_size

        return dx
