import numpy as np
import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread("obraz1.jpeg")
img2 = cv2.imread("obraz2.jpeg")
img3 = cv2.imread("obraz3.jpeg")

test = np.ones((3, 3, 3))
test[:, :, 1] = 2
test[:, :, 2] = 3
#print(test)

#img = test[:, :, 1].copy

#imgg = test[0, :, 0]

#print(test[0:2, 1:2, 0])

print(test.shape)
print(test.shape[0])
print(test.shape[1])

print(test[0::2, 0::2, 1])
'''
[2, 1, 2]
[3, 2, 3]
[2, 1, 2]
'''


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cfa(img):
    #img = image.copy()
    s = (img.shape[0], img.shape[1])
    out = np.zeros(s)
    out[0::2, 1::2] = img[0::2, 1::2, 0]
    out[1::2, 0::2] = img[1::2, 0::2, 2]
    out[0::2, 0::2] = img[0::2, 0::2, 1]
    out[1, 1] = img[1, 1, 1]
    #out[0::2, 0::2] = img[0::2, 0::2, 1]
    #out[1::2, 1::2]=img[1::2, 1::2]
    #out[1::2, 0::2] = img[1::2, 0::2, 2]
    #print(img[0::2, 1::2, 0])
    return out


x = cfa(test)
print(x)


plt.imshow(img2)
plt.show()
image2 = bgr2rgb(img2)
plt.imshow(image2)
plt.show()
imgg = cfa(image2)
plt.imshow(imgg)
plt.show()
