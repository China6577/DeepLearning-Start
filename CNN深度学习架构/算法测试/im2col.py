from CNN深度学习架构.神经网络层.util import im2col
import numpy as np

x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5,5, stride=1, pad=0)
print(col1.shape)
# print(col1)

x2 = np.array([[[[1,2,3],[4,5,6],[7,8,9]]]])
# 将特征图按照2*2的滤波器样子 横向展开
col2 = im2col(x2, 2, 2, stride=1, pad=0)
print(col2)
print(col2.shape)
