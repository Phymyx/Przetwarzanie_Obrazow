import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("PO_WB_set2/8D5U5525_C_A4.jpg")

plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.title("czysty obraz")


def white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    IL = lab[:, :, 0]
    Ia = lab[:, :, 1]
    Ib = lab[:, :, 2]
    average_a = np.average(Ia)
    average_b = np.average(Ib)
    k = 1.3
    lab[:, :, 1] = Ia - ((average_a - 128) * (IL / 255.0) * k)
    lab[:, :, 2] = Ib - ((average_b - 128) * (IL / 255.0) * k)
    lab[:, :, 0] = IL
    rgb_outp = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb_outp


plt.subplot(1, 2, 2)
img2 = white_balance(img1)
plt.imshow(img2)
plt.title("balans bieli k=1.3")
plt.show()
