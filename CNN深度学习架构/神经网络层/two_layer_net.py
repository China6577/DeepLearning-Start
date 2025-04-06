from collections import OrderedDict

from CNN深度学习架构.神经网络层.layers import *
from CNN深度学习架构.神经网络层.functions import calculate_gradient

import numpy as np


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化权重和偏置 生成层实现
        :param input_size: 输入的神经元数量
        :param hidden_size: 隐藏层的神经元数量
        :param output_size: 最终输出的神经元数量
        """
        # 初始化权重和偏置
        # 防止梯度消失和表现力受限: 根据使用的激活函数为Relu 权重的标准差为He初始值 根号(2/n) n为前一层的神经元个数
        # 当激活函数是sigmoid或tanh等S型曲线函数时 初始值用Xavier初始值 根号(1/n) n为前一层的神经元个数
        self.params = {}
        self.params['W1'] = np.sqrt(2/input_size) * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.sqrt(2/hidden_size) * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层实现
        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReluLayer()
        self.layers['Affine2'] = AffineLayer(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLossLayer()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        # x：输入数据  t：监督数据
        y = self.predict(x)
        y = np.argmax(y, axis=1)  # 返回最有可能的预测结果索引 一维数组
        if t.ndim != 1:
            t = t.argmax(axis=1)  # 返回正确结果的索引 一维数组
        accuracy = np.sum(y == t) / float(x.shape[0])  # 通过逐个判断两个一维数组的值是否相等来计算准确率

        return accuracy

    def backprop_gradient(self, x, t):
        """
        误差反向传播法求梯度 后面使用该方法求梯度 高效
        :param x: 输入数据
        :param t: 测试数据
        :return: 梯度矩阵
        """
        self.loss(x, t)

        dout = self.lastLayer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

    def numerical_gradient(self, x, t):
        """
        数值法求梯度 用于梯度确认
        :param x: 图像数据
        :param t: 正确解标签
        :return: 梯度矩阵
        """
        loss_W = lambda W: self.loss(x, t)  # loss_W是匿名函数W, 函数体是self.loss(x, t)

        grads = {}
        grads['W1'] = calculate_gradient(loss_W, self.params['W1'])
        grads['b1'] = calculate_gradient(loss_W, self.params['b1'])
        grads['W2'] = calculate_gradient(loss_W, self.params['W2'])
        grads['b2'] = calculate_gradient(loss_W, self.params['b2'])

        return grads



