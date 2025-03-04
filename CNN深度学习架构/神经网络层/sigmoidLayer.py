import numpy as np


class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        # sigmoid函数的反向传播的梯度公式为 dy = y(y-1)
        dx = dout * (1 - self.out) * self.out

        return dx
