import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.fftpack


def bgr2ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(int)


def ycrcb2rgb(img):
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb2ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def chroma_subsampling(img, wybor=False):
    if wybor == False:
        return img
    else:
        matrix8x4_2 = np.zeros((8, 4))
        for i in range(0, 8):
            z = 0
            for j in range(0, 8, 2):
                matrix8x4_2[i, z] = img[i, j]
                z += 1
    return matrix8x4_2


def chroma_resampling(img, wybor):
    if wybor == False:
        return img
    else:
        matrix8x8_2 = np.zeros((8, 8))
        for i in range(0, 8):
            z = 0
            for j in range(0, 8, 2):
                matrix8x8_2[i, j] = img[i, z]
                matrix8x8_2[i, j + 1] = img[i, z]
                z += 1
    return matrix8x8_2


def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')


def blokowanie_danych(img, y, x):
    matrix8x8 = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            matrix8x8[i, j] = img[i + y, j + x]
    return matrix8x8


def wstaw_blok(od, do, y, x):
    for i in range(8):
        for j in range(8):
            do[i + y, j + x] = od[i, j]
    return do


def zigzag(A):
    template = n = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ])

    if len(A.shape) == 1:
        B = np.zeros((8, 8))
        for r in range(0, 8):
            for c in range(0, 8):
                B[r, c] = A[template[r, c]]
    else:
        B = np.zeros((64,))
        for r in range(0, 8):
            for c in range(0, 8):
                B[template[r, c]] = A[r, c]
    return B


def kwantyzacja(xd, wybor):
    QY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 36, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])

    QC = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ])

    if wybor == True:
        qd = np.round(xd / QY).astype(int)
        return qd
    else:
        qd = np.round(xd / QC).astype(int)
        return qd


def dekwantyzacja(qd, wybor):
    QY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 36, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])

    QC = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ])

    if wybor == True:
        pd = qd * QY
        return pd
    else:
        pd = qd * QC
        return pd


def wykonaj(img, Y=True, chroma=False):
    width = img.shape[1]
    height = img.shape[0]
    matrix = np.zeros((height, width))
    if chroma == False:
        for y in range(0, height, 8):
            for x in range(0, width, 8):
                block = blokowanie_danych(img, y, x)
                dct = dct2(block)
                kwant = kwantyzacja(dct, Y)
                zz = zigzag(kwant)
                #
                zzoutp = zigzag(zz)
                dekwant = dekwantyzacja(zzoutp, Y)
                idct = idct2(dekwant)
                wstaw_blok(idct, matrix, y, x)
    else:
        for y in range(0, height, 8):
            for x in range(0, width, 16):
                block1 = blokowanie_danych(img, y, x)
                block2 = blokowanie_danych(img, y, x + 8)
                # chroma
                chrom1 = chroma_subsampling(block1, True)
                chrom2 = chroma_subsampling(block2, True)
                cos = np.hstack((chrom1, chrom2))
                dct = dct2(cos)
                kwant = kwantyzacja(dct, Y)
                zz = zigzag(kwant)
                #
                zzoutp = zigzag(zz)
                dekwant = dekwantyzacja(zzoutp, Y)
                idct = idct2(dekwant)
                # rechroma
                A = idct[:, :4]
                B = idct[:, 4:8]
                rechrom = chroma_resampling(A, True)
                rechrom1 = chroma_resampling(B, True)
                wstaw_blok(rechrom, matrix, y, x)
                wstaw_blok(rechrom1, matrix, y, x + 8)

    return matrix


def JPEG_compression(img, chroma=False):
    YCrCb = bgr2ycrcb(img)
    Y = YCrCb[:, :, 0]
    Cr = YCrCb[:, :, 1]
    Cb = YCrCb[:, :, 2]
    Y_2 = wykonaj(Y)
    Cr_2 = wykonaj(Cr, Y=False, chroma=chroma)
    Cb_2 = wykonaj(Cb, Y=False, chroma=chroma)
    YCrCb_2 = np.dstack([Y_2, Cr_2, Cb_2]).astype(np.uint8)
    RGB = ycrcb2rgb(YCrCb_2)
    return RGB


def crop(img):
    '''#lew
    y = 300
    x = 250
    h = 425
    w = 375'''

    '''#motyl
    y = 150
    x = 500
    h = 406
    w = 756'''

    # motyl2
    y = 150
    x = 500
    h = 278
    w = 628

    '''y = 300
    x = 250
    h = 425
    w = 375'''

    crop_image = img[y:h, x:w]
    #cv2.imshow("cropped", crop_image)
    #cv2.waitKey(0)
    return crop_image


def ploty(IMG, img):
    YCrCb1 = rgb2ycrcb(IMG)
    Y1 = YCrCb1[:, :, 0]
    Cr1 = YCrCb1[:, :, 1]
    Cb1 = YCrCb1[:, :, 2]
    fig, axs = plt.subplots(4, 2, sharey=True)
    fig.set_size_inches(9, 13)
    axs[0, 0].imshow(IMG)
    axs[1, 0].imshow(Y1, cmap=plt.cm.gray)
    axs[2, 0].imshow(Cr1, cmap=plt.cm.gray)
    axs[3, 0].imshow(Cb1, cmap=plt.cm.gray)

    YCrCb2 = rgb2ycrcb(img)
    Y2 = YCrCb2[:, :, 0]
    Cr2 = YCrCb2[:, :, 1]
    Cb2 = YCrCb2[:, :, 2]
    axs[0, 1].imshow(img)
    axs[1, 1].imshow(Y2, cmap=plt.cm.gray)
    axs[2, 1].imshow(Cr2, cmap=plt.cm.gray)
    axs[3, 1].imshow(Cb2, cmap=plt.cm.gray)


image = cv2.imread("creeper.png")

output = JPEG_compression(image)
output1 = JPEG_compression(image, chroma=True)

#crop1 = crop(output)
#crop2 = crop(output1)

'''#plt.subplot(1, 3, 1)
#image2 = bgr2rgb(image)
#plt.imshow(image2)
#plt.subplot(1, 2, 1)
plt.title("4:4:4")
plt.imshow(output)
plt.show()
#plt.subplot(1, 2, 2)
plt.title("4:2:2")
plt.imshow(output1)
plt.show()'''

ploty(output, output1)
#ploty(crop1, crop2)
plt.show()
