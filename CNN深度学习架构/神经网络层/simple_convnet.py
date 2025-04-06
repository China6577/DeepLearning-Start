import pickle
from collections import OrderedDict

from CNN深度学习架构.神经网络层.layers import *

class SimpleConvNet:
    """
    Convolutional Neural Network
    网络组成: Convolution - ReLU - Pooling - Affine - ReLU - Affine - Softmax
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        """
        :param input_dim: 输入数据的维度(通道 高 长) MNIST的情况下为784
        :param conv_param: 卷积层的超参数
        :param hidden_size: 隐藏层的神经元个数
        :param output_size: 输出层的神经元个数 MNIST的情况下为10
        :param weight_init_std: 初始化权重标准差
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]

        # 输入的形状大小
        input_size = input_dim[1]

        # 卷积层输出的大小
        conv_output_size = int((input_size - filter_size + 2 * filter_pad) / filter_stride + 1)

        # 以卷积层输出大小的一半为池化层的输入大小
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {}

        # 卷积层参数
        self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)

        # 全连接层1的参数
        self.params["W2"] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params["b2"] = np.zeros(hidden_size)

        # 全连接层2的参数
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        # 有序字典 可以迭代字典
        self.layers = OrderedDict()

        self.layers["Conv1"] = ConvolutionLayer(self.params["W1"], self.params["b1"], conv_param["stride"],
                                                conv_param["pad"])

        self.layers["Relu1"] = ReluLayer()
        self.layers["Pool1"] = PoolingLayer(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = AffineLayer(self.params["W2"], self.params["b2"])

        self.layers["Relu2"] = ReluLayer()
        self.layers["Affine2"] = AffineLayer(self.params["W3"], self.params["b3"])
        self.lastLayer = SoftmaxWithLossLayer()

    # 向前传播预测
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        # 向前传播
        self.loss(x, t)

        # 向后传播
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 学习 更新权重和偏置
        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Affine1"].dW
        grads["b2"] = self.layers["Affine1"].db
        grads["W3"] = self.layers["Affine2"].dW
        grads["b3"] = self.layers["Affine2"].db

        return grads

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    # 借助 pickle 模块 提供了对象序列化和反序列化的功能 能够把复杂的 Python 对象保存到文件中 或者在网络上进行传输
    # 可把程序运行过程中生成的对象保存到磁盘上，以便后续再次使用

    # 用于保存训练好的参数
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            # wb 以二进制形式写入
            # 将 params 对象序列化并写入到文件 params.pkl 中
            pickle.dump(params, f)

    # 用于加载训练好的参数
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]
