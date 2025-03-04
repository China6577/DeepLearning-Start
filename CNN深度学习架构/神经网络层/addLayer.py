class AddLayer:
    """
    加法层实现
    """

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = self.x + self.y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
