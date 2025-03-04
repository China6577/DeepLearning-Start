import numpy as np
from common import sigmoid, softmax, cross_entropy_error, numerical_gradient


# 简单的双层神经网络
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化权重和偏置
        :param input_size: 输入的神经元数量
        :param hidden_size: 隐藏层的神经元数量
        :param output_size: 最终输出的神经元数量
        :param weight_init_std: 权重初始化的标准差
        """

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """
        通过 y=wx+b 计算预测结果
        :param x: 图像数据
        :return:预测结果
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """
        计算误差 计算的数值越小越好
        :param x: 图像数据
        :param t: 正确解标签
        :return: 误差
        """
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """
        计算准确率 越接近1越好
        :param x: 图像数据
        :param t: 正确解标签
        :return:准确率
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)  # 每一行中的最大值的索引位置，并将这些索引组成一个新的一维数组赋值给y
        t = np.argmax(t, axis=1)  # 每一行中的最大值的索引位置，并将这些索引组成一个新的一维数组赋值给t

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        使用'梯度下降法'学习调整参数
        :param x: 图像数据
        :param t: 正确解标签
        :return: 学习一次后的参数
        """
        loss_W = lambda W: self.loss(x, t)  # loss_W是匿名函数W, 函数体是self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
