import numpy as np


class SGD:
    """
    梯度下降法(Stochastic Gradient Descent)最优化算法:
        新参数 = 旧参数 - 旧参数梯度 * 学习率
    缺点: 梯度方向有时候并不会指向最小值方向
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    动量(Momentum)最优化算法:
        速度v =  动量momentum * 速度v - 旧参数梯度 * 学习率
        新参数 = 旧参数 + 速度v
    优点: 当旧参数梯度为0或接近0时 新参数依然会有变化
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """
    学习率衰减(Adaptive Grad)法:
        h = h + 旧参数梯度 * 旧参数梯度
        新参数 = 旧参数 - [学习率 * (1/根号h)] * 旧参数梯度
    优点: h存储了所有旧参数梯度的平方和来降低学习率 使得学习越深入学习的幅度越小 更好地去接近最值
    缺点: 无止境地学习也会导数值h趋于0 更新终止
    """

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= (self.lr / (np.sqrt(self.h[key]) + 1e-7)) * grads[key]


class RMSProp:
    """
    均方根传播(Root Mean Square Propagation)优化算法:
        会逐渐的遗忘过去的梯度信息并学习新的梯度信息
    优点: 自适应学习率 解决AdaGrad学习率单调递减的问题‌
    """

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate  # 衰变率
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate  # 遗忘先前的部分梯度信息
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]  # 学习新的梯度信息
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    """
    Momentum和AdaGrad方法的结合
    代码部分是借鉴https://arxiv.org/abs/1412.6980v8
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
