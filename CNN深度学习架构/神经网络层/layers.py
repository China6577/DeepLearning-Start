from CNN深度学习架构.神经网络层.functions import *
import numpy as np
from CNN深度学习架构.神经网络层.util import im2col, col2im


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

        # self.dW 和 self.db 可以用于后续的参数更新
        self.dW = None
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


class BatchNormalizationLayer:
    """
    可参考: https://arxiv.org/abs/1502.03167
    设置在Affine层和Relu层之间
    批量归一化层:
        1.对于每个神经元的输出 计算该神经元在当前小批量中的均值和方差
        2.使用计算得到的均值和方差对数据进行标准化 转换为均值为0 方差为1的数据集
        3.引入可学习的参数进行缩放和平移，以恢复原始数据的分布
    作用: 在训练神经网络时 每一层的数据分布会随着训练的进行而发生变化 这可能导致训练过程变得不稳定
         Batch Normalization将数据重新调整为标准正态分布，使得训练过程更加平稳
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class DropoutLayer:
    """
    丢弃层:
        1.学习时随机删除删除神经元 相当于不同的模型在训练
        2.推理时通过神经元输出乘目前神经元占比(1-删除比例) 可以得到模型的平均值
    作用: 正则化 为了抑制过拟合
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio  # 每次删除神经元的比例
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:  # 训练模式(学习)
            # self.mask会随机生成和x形状相同的数组 并将值比dropout_ratio大的元素设为True 以 False 的形式保存要删除的神经元
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio  # *是获得x多个维度的数据 从而生成和x一样大小的[0,1)数组
            return x * self.mask  # 训练时将丢弃后的神经元送到下一层
        else:  # 推理模式
            return x * (1.0 - self.dropout_ratio)  # 因为训练时丢弃了dropout_ratio比例的神经元进行学习
            # 所以推理时要乘上(1-dropout_ratio)计算均值来和训练时的数据多少保持一致 传到下一层来预测

    def backward(self, dout):
        return dout * self.mask  # True则不变 False则输出0(因为神经元被丢弃)


class ConvolutionLayer:
    """
    卷积层:
        相当于图形处理的 “滤波器运算”
        保持训练时数据形状不变
    """

    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape  # 滤波器的形状
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)  # 卷积运算后输出的高
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)  # 卷积运算后输出的宽

        # im2col用于加快多维矩阵的计算 将多维矩阵转换为二维矩阵
        col = im2col(x, FH, FW, self.stride, self.pad)  # 特征图横向展开
        col_W = self.W.reshape(FN, -1).T  # 滤波器纵向展开 形状为 (滤波器个数 * 滤波器元素个数) 不能写成  self.W.reshape(-1, FN)
        out = np.dot(col, col_W) + self.b

        # (N, out_h, out_w, -1).transpose(0, 3, 1, 2) 将第二个位置变为C((N, out_h, out_w, -1)索引为3的值)...
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  # 将输出变为(N, C, H, W)的形式

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class PoolingLayer:
    """
    池化层:
        使用Max池化(取目标区域的最大值)
        池化层的步幅和窗口大小相同 无填充
    """

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)  # 池化层输出的高
        out_w = int(1 + (W - self.pool_w) / self.stride)  # 池化层输出的宽

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)

        # 每一行为池化目标区域的值组成的数组
        col = col.reshape(-1, self.pool_h * self.pool_w)  # 数据转化为 (池化层元素数组个数 * 池化层数组元素个数) 的形状

        arg_max = np.argmax(col, axis=1)

        col = np.max(col, axis=1)  # 求沿着列方向对每一行(单个池化层元素数组)的最大值

        out = col.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
