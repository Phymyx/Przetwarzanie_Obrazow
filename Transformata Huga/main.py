import numpy as np
import matplotlib.pyplot as plt
import cv2


def szarosc(img):
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    #Y2 = 0.299 * R + 0.7152 * G + 0.114 * B
    return Y


def okregi():
    r = 15
    rad = np.linspace(-np.pi, np.pi, 500)
    x = np.round(r * np.sin(rad) + r + 1).astype(int)
    y = np.round(r * np.cos(rad) + r + 1).astype(int)

    maska = np.zeros((2*r + 3, 2*r + 3))

    for k in range(500):
        maska[y[k], x[k]] = 1

    '''plt.imshow(maska)
    plt.show()

    plt.plot(x, y)
    plt.axis('equal')
    plt.show()'''
    return maska


image = cv2.imread("HT.png")
gray = szarosc(image)
'''plt.imshow(gray, cmap="gray")
plt.show()'''

okregi()
