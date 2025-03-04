class ReluLayer:
    """
    relu层(以relu作为激活函数)：
        1.当x小于等于0则输出0
        2.x大于0则输出x
    """

    def __init__(self):
        self.mask = None  # mask为实例变量 由forward函数和backward函数共享

    def forward(self, x):
        """
        relu函数的实现
        :param x: 输入的矩阵
        :return: 输入矩阵通过relu正向传播后的矩阵
        """
        self.mask = (x <= 0)  # mask由布尔值构成的numpy数组（矩阵） 当x小于等于0是为ture
        out = x.copy()
        out[self.mask] = 0  # 为ture则设置为0 false则不变

        return out

    def backward(self, dout):
        """
        反向传播法求梯度
        :param dout: 从后续层传来的梯度矩阵
        :return: 梯度矩阵通过relu反向传播后的梯度矩阵
        """
        dout[self.mask] = 0  # 如果x小于等于0则梯度值为0
        dx = dout  # x等于等于0则梯度值不变

        return dx
