import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("PO_WB_set2/8D5U5525_C_A4.jpg")
img2 = cv2.imread("PO_WB_set2/8D5U5525_D_A4.jpg")
img3 = cv2.imread("PO_WB_set2/8D5U5525_F_A4.jpg")
img4 = cv2.imread("PO_WB_set2/8D5U5525_S_A4.jpg")
img5 = cv2.imread("PO_WB_set2/8D5U5525_T_A4.jpg")


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def white_balance(img, k):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    IL = lab[:, :, 0]
    Ia = lab[:, :, 1]
    Ib = lab[:, :, 2]
    average_a = np.average(Ia)
    average_b = np.average(Ib)
    lab[:, :, 1] = Ia - ((average_a - 128) * (IL / 255.0) * k)
    lab[:, :, 2] = Ib - ((average_b - 128) * (IL / 255.0) * k)
    lab[:, :, 0] = IL
    rgb_outp = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb_outp


def plots(img):
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("czysty obraz")

    plt.subplot(2, 2, 2)
    img_white = white_balance(img, 0.8)
    plt.imshow(img_white)
    plt.title("balans bieli k=0.8")

    plt.subplot(2, 2, 3)
    img_white2 = white_balance(img, 1.3)
    plt.imshow(img_white2)
    plt.title("balans bieli k=1.3")

    plt.subplot(2, 2, 4)
    img_white3 = white_balance(img, 2.0)
    plt.imshow(img_white3)
    plt.title("balans bieli k=2.0")

    plt.show()


img1rgb = bgr2rgb(img1)
img2rgb = bgr2rgb(img2)
img3rgb = bgr2rgb(img3)
img4rgb = bgr2rgb(img4)
img5rgb = bgr2rgb(img5)

plots(img1rgb)
plots(img2rgb)
plots(img3rgb)
plots(img4rgb)
plots(img5rgb)
