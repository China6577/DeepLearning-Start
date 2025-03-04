from two_layer_net import TwoLayerNet
import numpy as np

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

W1 = net.params['W1']
b1 = net.params['b1']
W2 = net.params['W2']
b2 = net.params['b2']
print(W1.shape, W2.shape, b1.shape, b2.shape)

x = np.random.rand(100, 784)  # 输入大小为100x784 批的大小为100 784为神经元个数
t = np.random.rand(100, 10)  # 最终大小为100x10 批的大小为100 10为输出层神经元个数
y = net.predict(x)
