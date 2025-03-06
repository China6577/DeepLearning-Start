import numpy as np
from mnist数据集.mnist import load_mnist
from CNN深度学习架构.神经网络层.twoLayerNet import TwoLayerNet
from CNN深度学习架构.神经网络层.optimizer import *

# 加载训练数据和测试数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 构建神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 学习次数
train_size = x_train.shape[0]  # 训练样本的数量
batch_size = 100  # mini-batch训练法一次训练的样本个数
learning_rate = 0.1  # 学习率用于梯度下降法学习参数
train_loss_list = []  # 每学习一个batch的学习误差
train_acc_list = []  # 每学习完一次所有训练数据 测试在训练数据集中的准确率
test_acc_list = []  # 每学习完一次所有训练数据 测试在测试数据集中的准确率

iter_per_epoch = max(train_size / batch_size, 1)  # 训练完一次所有数据的次数

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)  # 从0到train_size-1中挑选batch_size个 作为数据选取的索引数组
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 获得每个参数的梯度
    grad = network.backprop_gradient(x_batch, t_batch)

    # 参数的学习 可以使用CNN深度学习架构.神经网络层.optimizer中的几种算法学习 只需要更改类名即可
    optimizer = Adam()
    optimizer.update(network.params, grad)

    loss = network.loss(x_train, t_train)
    train_loss_list.append(loss)
    print(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
