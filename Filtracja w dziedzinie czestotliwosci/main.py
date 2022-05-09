import numpy as np
import matplotlib.pyplot as plt
import cv2


def mask():
    
    return


image = cv2.imread("Lenna.png", 0)

image_float = image / 255

widmo = np.fft.fft2(image_float)
widmo = np.abs(widmo)

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")

plt.subplot(2, 3, 2)
plt.title("widmo <0, 255>")
plt.imshow(widmo, vmax=255)

widmo2 = np.fft.fftshift(widmo)

plt.subplot(2, 3, 3)
plt.title("widmo po fftshift <0, 255>")
plt.imshow(widmo2, vmax=255)

plt.subplot(2, 3, 5)
plt.title("widmo <0, max>")
plt.imshow(widmo)

widmo2 = np.fft.fftshift(widmo)

plt.subplot(2, 3, 6)
plt.title("widmo po fftshift <0, max>")
plt.imshow(widmo2)
plt.show()
