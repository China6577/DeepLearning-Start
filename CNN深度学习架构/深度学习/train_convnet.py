import time
import matplotlib.pyplot as plt
import numpy as np

from mnist数据集.mnist import load_mnist
from CNN深度学习架构.神经网络层.simple_convnet import SimpleConvNet
from CNN深度学习架构.神经网络层.trainer import Trainer
from pathlib import Path

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 处理花费时间较长的情况下减少数据
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 3

network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 判断是否存在训练参数 有则加载并使用
file_path = Path('params.pkl')
if file_path.exists():
    network.load_params()

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000, verbose=True)
start_time = time.time()

trainer.train()

end_time = time.time()
time = round(end_time - start_time)

# 保存参数
network.save_params("params.pkl")
print("成功保存学习后的参数!")
print(f"训练共花费 {time} s")

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(0,max_epochs+1)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()