import numpy as np
from mnist数据集.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

if __name__ == '__main__':
    img = x_train[0]
    print(img.shape)
    label = t_train[0]

    img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸

    print(img.shape)

    img_show(img)
