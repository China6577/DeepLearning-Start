import numpy as np
from collections import OrderedDict

from affineLayer import AffineLayer
from reluLayer import ReluLayer
from softmaxWithLossLayer import SoftmaxWithLossLayer


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化权重和偏置 生成层实现
        :param input_size: 输入的神经元数量
        :param hidden_size: 隐藏层的神经元数量
        :param output_size: 最终输出的神经元数量
        :param weight_init_std: 权重初始化的标准差
        """
        # 初始化权重和偏置
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层实现
        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReluLayer()
        self.layers['Affine2'] = AffineLayer(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLossLayer

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        # x：输入数据  t：监督数据
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t=t.argmax(axis=1)
        accuracy = np.sum(y==t) / float(x.shape[0])

        return accuracy