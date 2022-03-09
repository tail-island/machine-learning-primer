import cv2
import matplotlib.pyplot as plot
import numpy as np


def filter(image, kernel):
    result = np.zeros((np.shape(image)[0] - 2, np.shape(image)[1] - 2))

    for y in range(np.shape(result)[0]):
        for x in range(np.shape(result)[1]):
            # 画像の該当部分とカーネルのベクトルの内積を求めて新たな画像を作成します。
            result[y, x] = np.inner(image[y: y + 3, x: x + 3].flatten(), kernel.flatten())

    result[result <   0] =   0  # noqa: E222
    result[result > 255] = 255

    return result


image = cv2.cvtColor(cv2.imread('4.2.07.tiff'), cv2.COLOR_RGB2GRAY)

plot.imshow(image)
plot.show()

plot.imshow(filter(image, np.array([[0, 2, 0], [2, -8, 2], [0, 2, 0]])))
plot.show()
