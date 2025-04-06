import random
from pathlib import Path

from CNN深度学习架构.神经网络层.functions import softmax
from CNN深度学习架构.神经网络层.simple_convnet import SimpleConvNet
from mnist数据集.mnist import load_mnist
from CNN深度学习架构.算法测试.show_image import img_show

# 读入数据 不展平
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=False)

random_index = random.randint(0, x_test.shape[0] - 1)

network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 判断是否存在训练参数 有则加载并使用
if Path('../深度学习/params.pkl'):
    network.load_params('../深度学习/params.pkl')
else:
    print("模型暂未训练")
    exit(0)

img_show(x_test[random_index].reshape(28,28))

# argmax返回每一行最大值索引
pred_val = softmax(network.predict(x_test[random_index].reshape(1,1,28,28))).argmax(axis=1)

print(pred_val[0])
print(t_test[random_index])