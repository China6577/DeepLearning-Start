import numpy as np

from mnist数据集.mnist import load_mnist
from CNN深度学习架构.神经网络层.twoLayerNet import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.backprop_gradient(x_batch, t_batch)

# 误差很小说明误差反向传播法计算结果正确
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))  #权重和偏置是一个矩阵 所以可以通过求平均值来看和数值法求导的误差
    print(key + ": " + str(diff))  # diff是numpy.float64位数据要用str函数转换为字符串才能拼接
