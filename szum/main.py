import numpy as np
import random
from scipy.stats import erlang
from math import log10, sqrt
import cv2
import matplotlib.pyplot as plt


def GaussianNoise(img, alpha):
    gauss = np.random.normal(0, 25, (img.shape))
    noisy = (img + alpha * gauss).clip(0, 255).astype(np.uint8)
    return noisy


def ValsNoise(img, alpha):
    vals = len(np.unique(img))
    vals = alpha * 2 ** np.ceil(np.log2(vals))
    noisy = (np.random.poisson(img * vals) / float(vals)).clip(0, 255).astype(np.uint8)
    return noisy


def RandomNoise(img, alpha):
    rand = 25 * np.random.random((img.shape))
    noisy = (img + alpha * rand).clip(0, 255).astype(np.uint8)
    return noisy


def noise_SnP(img, S=255, P=0, rnd=(333, 9999)):
    r, c = img.shape
    number_of_pixels = random.randint(rnd[0], rnd[1])
    for i in range(number_of_pixels):
        y = random.randint(0, r - 1)
        x = random.randint(0, c - 1)
        img[y][x] = S
    number_of_pixels = random.randint(rnd[0], rnd[1])
    for i in range(number_of_pixels):
        y = random.randint(0, r - 1)
        x = random.randint(0, c - 1)
        img[y][x] = P
    return img


def ErlangNoise():
    numargs = erlang.numargs
    [a] = [0.6, ] * numargs
    rv = erlang(a)

    print("RV : \n", rv)
    R = erlang.rvs(a, scale=2, size=10)
    print("Random Variates : \n", R)
    return R


def arytmetyczny(img):
    h, w = img.shape
    suma = 0
    for y in range(h):
        for x in range(w):
            suma += img[y, x]
    filtr = (1 / (h * w)) * suma
    return filtr


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def change(img):
    img = img[:, :, 1]
    for x in range(100):
        img[0, x] = img[0, x] + 0.5
        img[100, x] = img[100, x] - 0.4
    return img


image = cv2.imread("lenna.png")
#print(f"PSNR value is {value} dB")

R = image[:, :, 0]
G = image[:, :, 1]
B = image[:, :, 2]

vec = []
X = []

for x in range(255):
    gaus = GaussianNoise(image, x)
    ar = arytmetyczny(gaus)
    p = PSNR(image, ar)
    vec.append(p)
    X.append(x)


plt.plot(X, vec)
plt.show()
