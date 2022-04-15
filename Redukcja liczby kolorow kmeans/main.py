import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

image = cv2.imread("tucan.jpeg")

#predict z kmeans zwroci nam wartosc klastra

test = np.ones((3, 3))

def Jasnosc(img):
    x = np.mean(img)
    return x


def Kontrast(img, J):
    y = (np.sqrt(np.mean(np.power(img - J, 2))))
    return y


def JK(img):
    J = np.mean(img)
    K = (np.sqrt(np.mean(np.power(img - J, 2))))
    arr = np.array([J, K])
    return arr

def JKRGB(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.144 * B
    R1 = JK(R)
    R2 = JK(G)
    R3 = JK(B)
    Y1 = JK(Y)
    RGB = np.array([R1, R2, R3, Y1])
    #out = np.rot90(RGB)
    return RGB


J = Jasnosc(test)
print(J)
K = Kontrast(test, J)
print(K)
A1 = JK(test)
print(A1)
RGB = JKRGB(image)
print(RGB)

def reduction():

