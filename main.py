# from tkinter import *
from math import *
from tkinter.filedialog import *

import numpy as np
import scipy
from PIL import ImageTk, Image
from cv2 import *
from scipy import ndimage
from scipy import signal

np.set_printoptions(threshold=np.inf, linewidth=850)


# import cv2.cv2 as cv


def choosePic():
    global img
    img_path = askopenfilename(initialdir='./DB3_B/', title='选择待识别图片',
                               filetypes=[("tif", "*.tif"), ("jpg", "*.jpg"), ("png", "*.png")])
    if img_path:
        print(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        print(type(img))
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图

        rows, cols = np.shape(img)
        aspect_ratio = np.double(rows) / np.double(cols)

        new_rows = 200  # randomly selected number
        new_cols = new_rows / aspect_ratio

        img = cv2.resize(img, (int(new_cols), int(new_rows)))
        show(img, oriImg)


def image_enhance():
    global img
    blksze = 16
    thresh = 0.1
    normim, mask = ridge_segment(img, blksze, thresh)  # normalise the image and find a ROI
    # imshow("norm", normim)

    gradientsigma = 1
    blocksigma = 7
    orientsmoothsigma = 7
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma)  # find orientation of every pixel
    # imshow("orient", orientim)

    blksze = 38
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freq, medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,
                               maxWaveLength)  # find the overall frequency of ridges
    # imshow("freq", freq)

    freq = medfreq * mask
    kx = 0.65
    ky = 0.65
    newim = ridge_filter(normim, orientim, freq, kx, ky)  # create gabor filter and do the actual filtering
    # imshow("new",newim)

    img = 255 * (newim >= -3)
    # print(img)
    show(img, enhanceImg)


def normalise(img):
    normed = (img - np.mean(img)) / (np.std(img))
    return normed


def ridge_segment(im, blksze, thresh):  # img,16,0.1

    rows, cols = im.shape

    im = normalise(im)  # normalise to get zero mean and unit standard deviation 归一化？
    # imshow("norm",im)

    new_rows = np.int(blksze * np.ceil((np.float(rows)) / (np.float(blksze))))
    new_cols = np.int(blksze * np.ceil((np.float(cols)) / (np.float(blksze))))

    padded_img = np.zeros((new_rows, new_cols))
    stddevim = np.zeros((new_rows, new_cols))

    padded_img[0:rows][:, 0:cols] = im

    for i in range(0, new_rows, blksze):
        for j in range(0, new_cols, blksze):
            block = padded_img[i:i + blksze][:, j:j + blksze]

            stddevim[i:i + blksze][:, j:j + blksze] = np.std(block) * np.ones(block.shape)

    stddevim = stddevim[0:rows][:, 0:cols]

    mask = stddevim > thresh

    mean_val = np.mean(im[mask])

    std_val = np.std(im[mask])

    normim = (im - mean_val) / (std_val)
    # imshow("norm",normim)

    return (normim, mask)


def ridge_orient(im, gradientsigma, blocksigma, orientsmoothsigma):
    rows, cols = im.shape
    # Calculate image gradients.
    sze = np.fix(6 * gradientsigma)
    if np.remainder(sze, 2) == 0:
        sze = sze + 1

    gauss = cv2.getGaussianKernel(np.int(sze), gradientsigma)
    f = gauss * gauss.T

    fy, fx = np.gradient(f)  # Gradient of Gaussian

    # Gx = ndimage.convolve(np.double(im),fx);
    # Gy = ndimage.convolve(np.double(im),fy);

    Gx = signal.convolve2d(im, fx, mode='same')
    Gy = signal.convolve2d(im, fy, mode='same')

    Gxx = np.power(Gx, 2)
    Gyy = np.power(Gy, 2)
    Gxy = Gx * Gy

    # Now smooth the covariance data to perform a weighted summation of the data.

    sze = np.fix(6 * blocksigma)

    gauss = cv2.getGaussianKernel(np.int(sze), blocksigma)
    f = gauss * gauss.T

    Gxx = ndimage.convolve(Gxx, f)
    Gyy = ndimage.convolve(Gyy, f)
    Gxy = 2 * ndimage.convolve(Gxy, f)

    # Analytic solution of principal direction
    denom = np.sqrt(np.power(Gxy, 2) + np.power((Gxx - Gyy), 2)) + np.finfo(float).eps

    sin2theta = Gxy / denom  # Sine and cosine of doubled angles
    cos2theta = (Gxx - Gyy) / denom

    if orientsmoothsigma:
        sze = np.fix(6 * orientsmoothsigma)
        if np.remainder(sze, 2) == 0:
            sze = sze + 1
        gauss = cv2.getGaussianKernel(np.int(sze), orientsmoothsigma)
        f = gauss * gauss.T
        cos2theta = ndimage.convolve(cos2theta, f)  # Smoothed sine and cosine of
        sin2theta = ndimage.convolve(sin2theta, f)  # doubled angles

    orientim = np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2
    return orientim


def ridge_freq(im, mask, orient, blksze, windsze, minWaveLength, maxWaveLength):
    rows, cols = im.shape
    freq = np.zeros((rows, cols))

    for r in range(0, rows - blksze, blksze):
        for c in range(0, cols - blksze, blksze):
            blkim = im[r:r + blksze][:, c:c + blksze]
            blkor = orient[r:r + blksze][:, c:c + blksze]

            freq[r:r + blksze][:, c:c + blksze] = frequest(blkim, blkor, windsze, minWaveLength, maxWaveLength)

    freq = freq * mask
    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    non_zero_elems_in_freq = freq_1d[0][ind]

    meanfreq = np.mean(non_zero_elems_in_freq)
    medianfreq = np.median(non_zero_elems_in_freq)  # does not work properly
    return freq, meanfreq


def frequest(im, orientim, windsze, minWaveLength, maxWaveLength):
    rows, cols = np.shape(im)

    # Find mean orientation within the block. This is done by averaging the
    # sines and cosines of the doubled angles before reconstructing the
    # angle again.  This avoids wraparound problems at the origin.

    cosorient = np.mean(np.cos(2 * orientim))
    sinorient = np.mean(np.sin(2 * orientim))
    orient = atan2(sinorient, cosorient) / 2

    # Rotate the image block so that the ridges are vertical

    # ROT_mat = cv2.getRotationMatrix2D((cols/2,rows/2),orient/np.pi*180 + 90,1)
    # rotim = cv2.warpAffine(im,ROT_mat,(cols,rows))
    rotim = scipy.ndimage.rotate(im, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode='nearest')

    # Now crop the image so that the rotated image does not contain any
    # invalid regions.  This prevents the projection down the columns
    # from being mucked up.

    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]

    # Sum down the columns to get a projection of the grey values down
    # the ridges.

    proj = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(proj, windsze, structure=np.ones(windsze))

    temp = np.abs(dilation - proj)

    peak_thresh = 2

    maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
    maxind = np.where(maxpts)

    rows_maxind, cols_maxind = np.shape(maxind)

    # Determine the spatial frequency of the ridges by divinding the
    # distance between the 1st and last peaks by the (No of peaks-1). If no
    # peaks are detected, or the wavelength is outside the allowed bounds,
    # the frequency image is set to 0

    if cols_maxind < 2:
        freqim = np.zeros(im.shape)
    else:
        NoOfPeaks = cols_maxind
        waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1)
        if waveLength >= minWaveLength and waveLength <= maxWaveLength:
            freqim = 1 / np.double(waveLength) * np.ones(im.shape)
        else:
            freqim = np.zeros(im.shape)

    return freqim


def ridge_filter(im, orient, freq, kx, ky):
    angleInc = 3
    im = np.double(im)
    rows, cols = im.shape
    newim = np.zeros((rows, cols))

    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.

    non_zero_elems_in_freq = freq_1d[0][ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

    unfreq = np.unique(non_zero_elems_in_freq)

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.

    sigmax = 1 / unfreq[0] * kx
    sigmay = 1 / unfreq[0] * ky

    sze = np.int(np.round(3 * np.max([sigmax, sigmay])))

    x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))

    reffilter = np.exp(-(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(
        2 * np.pi * unfreq[0] * x)  # this is the original gabor filter

    filt_rows, filt_cols = reffilter.shape

    angleRange = np.int(180 / angleInc)

    gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))

    for o in range(0, angleRange):
        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation *along* the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.

        rot_filt = scipy.ndimage.rotate(reffilter, -(o * angleInc + 90), reshape=False)
        gabor_filter[o] = rot_filt

    # Find indices of matrix points greater than maxsze from the image
    # boundary

    maxsze = int(sze)

    temp = freq > 0
    validr, validc = np.where(temp)

    temp1 = validr > maxsze
    temp2 = validr < rows - maxsze
    temp3 = validc > maxsze
    temp4 = validc < cols - maxsze

    final_temp = temp1 & temp2 & temp3 & temp4

    finalind = np.where(final_temp)

    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)

    maxorientindex = np.round(180 / angleInc)
    orientindex = np.round(orient / np.pi * 180 / angleInc)

    # do the filtering

    for i in range(0, rows):
        for j in range(0, cols):
            if orientindex[i][j] < 1:
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if orientindex[i][j] > maxorientindex:
                orientindex[i][j] = orientindex[i][j] - maxorientindex
    finalind_rows, finalind_cols = np.shape(finalind)
    sze = int(sze)
    for k in range(0, finalind_cols):
        r = validr[finalind[0][k]]
        c = validc[finalind[0][k]]

        img_block = im[r - sze:r + sze + 1][:, c - sze:c + sze + 1]

        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

    return newim


def VThin(image, array):
    # h = image.height
    # w = image.width
    h, w = image.shape
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i, j - 1] + image[i, j] + image[i, j + 1] if 0 < j < w - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def HThin(image, array):
    # h = image.height
    # w = image.width
    h, w = image.shape
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i - 1, j] + image[i, j] + image[i + 1, j] if 0 < i < h - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def thinning(num=10):
    # iXihua = cv.CreateImage(cv.GetSize(image), 8, 1)
    # cv.Copy(image, iXihua)
    global img
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

    # img = img.tolist()
    # img.copyTo(iXihua)
    for i in range(num):
        VThin(img, array)
        HThin(img, array)
    show(img, thinningImg)
    print(img)


def feature():
    global img
    # endpoint1 = img
    # endpoint2 = img
    features = []
    # endpoint = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img[i, j] == 0:  # 像素点为黑
                m = i
                n = j

                eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1], img[m, n + 1],
                              img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]

                if sum(eightField) / 255 == 7:  # 黑色块1个，端点

                    # 判断是否为指纹图像边缘
                    if sum(img[:i, j]) == 255 * i or sum(img[i + 1:, j]) == 255 * (w - i - 1) or sum(
                            img[i, :j]) == 255 * j or sum(img[i, j + 1:]) == 255 * (h - j - 1):
                        continue
                    canContinue = TRUE
                    # print(m, n)
                    coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1], [m + 1, n - 1],
                                  [m + 1, n], [m + 1, n + 1]]
                    for o in range(8):  # 寻找相连接的下一个点
                        if eightField[o] == 0:
                            index = o
                            m = coordinate[o][0]
                            n = coordinate[o][1]
                            # print(m, n, index)
                            break
                    # print(m, n, index)
                    for k in range(4):
                        coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                      [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                        eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1], img[m, n + 1],
                                      img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                        if sum(eightField) / 255 == 6:  # 连接点
                            for o in range(8):
                                if eightField[o] == 0 and o != 7 - index:
                                    index = o
                                    m = coordinate[o][0]
                                    n = coordinate[o][1]
                                    # print(m, n, index)
                                    break
                        else:
                            # print("false", i, j)
                            canContinue = FALSE
                    if canContinue:

                        if n - j != 0:
                            if i - m >= 0 and j - n > 0:
                                direction = atan((i - m) / (n - j)) + pi
                            elif i - m < 0 and j - n > 0:
                                direction = atan((i - m) / (n - j)) - pi
                            else:
                                direction = atan((i - m) / (n - j))
                        else:
                            if i - m >= 0:
                                direction = pi / 2
                            else:
                                direction = -pi / 2
                        feature = []
                        feature.append(i)
                        feature.append(j)
                        feature.append("endpoint")
                        feature.append(direction)
                        features.append(feature)

                elif sum(eightField) / 255 == 5:  # 黑色块3个，分叉点
                    coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1], [m + 1, n - 1],
                                  [m + 1, n], [m + 1, n + 1]]
                    junctionCoordinates = []
                    junctions = []
                    canContinue = TRUE
                    # 筛除不符合的分叉点
                    for o in range(8):  # 寻找相连接的下一个点
                        if eightField[o] == 0:
                            junctions.append(o)
                            junctionCoordinates.append(coordinate[o])
                    for k in range(3):
                        if k == 0:
                            a = junctions[0]
                            b = junctions[1]
                        elif k == 1:
                            a = junctions[1]
                            b = junctions[2]
                        else:
                            a = junctions[0]
                            b = junctions[2]
                        if (a == 0 and b == 1) or (a == 1 and b == 2) or (a == 2 and b == 4) or (a == 4 and b == 7) or (
                                a == 6 and b == 7) or (a == 5 and b == 6) or (a == 3 and b == 5) or (a == 0 and b == 3):
                            canContinue = FALSE
                            break

                    if canContinue:  # 合格分叉点
                        # print(junctions)
                        print(junctionCoordinates)
                        print(i, j, "合格分叉点")
                        directions = []
                        canContinue = TRUE
                        for k in range(3):  # 分三路进行
                            if canContinue:
                                junctionCoordinate = junctionCoordinates[k]
                                m = junctionCoordinate[0]
                                n = junctionCoordinate[1]
                                print(m, n, "start")
                                eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1],
                                              img[m, n + 1],
                                              img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                                coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                              [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                                canContinue = FALSE
                                for o in range(8):
                                    if eightField[o] == 0:
                                        a = coordinate[o][0]
                                        b = coordinate[o][1]
                                        print("a=", a, "b=", b)
                                        # print("i=", i, "j=", j)
                                        if (a != i or b != j) and (
                                                a != junctionCoordinates[0][0] or b != junctionCoordinates[0][1]) and (
                                                a != junctionCoordinates[1][0] or b != junctionCoordinates[1][1]) and (
                                                a != junctionCoordinates[2][0] or b != junctionCoordinates[2][1]):
                                            index = o
                                            m = a
                                            n = b
                                            canContinue = TRUE
                                            print(m, n, index, "支路", k)
                                            break
                                if canContinue:  # 能够找到第二个支路点
                                    for p in range(3):
                                        coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1],
                                                      [m, n + 1],
                                                      [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                                        eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1],
                                                      img[m, n - 1],
                                                      img[m, n + 1],
                                                      img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                                        if sum(eightField) / 255 == 6:  # 连接点
                                            for o in range(8):
                                                if eightField[o] == 0 and o != 7 - index:
                                                    index = o
                                                    m = coordinate[o][0]
                                                    n = coordinate[o][1]
                                                    print(m, n, index, "支路尾")
                                                    # print(m, n, index)
                                                    break
                                        else:
                                            # print("false", i, j)
                                            canContinue = FALSE
                                if canContinue:  # 能够找到3个连接点

                                    if n - j != 0:
                                        if i - m >= 0 and j - n > 0:
                                            direction = atan((i - m) / (n - j)) + pi
                                        elif i - m < 0 and j - n > 0:
                                            direction = atan((i - m) / (n - j)) - pi
                                        else:
                                            direction = atan((i - m) / (n - j))
                                    else:
                                        if i - m >= 0:
                                            direction = pi / 2
                                        else:
                                            direction = -pi / 2
                                    # print(direction)
                                    directions.append(direction)
                        if canContinue:
                            feature = []
                            feature.append(i)
                            feature.append(j)
                            feature.append("bifurcation")
                            feature.append(directions)
                            features.append(feature)
    print(features)
    for i in range(len(features)):
        txtFeature.insert(END, str(features[i]) + '\n')
    for m in range(len(features)):
        if features[m][2] == "endpoint":
            cv2.circle(img, (features[m][1], features[m][0]), 3, (0, 0, 255), 1)
        else:
            cv2.circle(img, (features[m][1], features[m][0]), 3, (0, 0, 255), -1)

    show(img, featureImg)


# def endpoint():
#     global img
#     endpoint1 = img
#     endpoint2 = img
#     endpoints = []
#     # endpoint = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     h, w = img.shape
#     for i in range(1, h - 1):
#         for j in range(1, w - 1):
#             if img[i, j] == 0:
#                 eightField = int(
#                     (img[i - 1, j - 1] + img[i - 1, j] + img[i - 1, j + 1] + img[i, j - 1] + img[i, j + 1] + \
#                      img[i + 1, j - 1] + img[i + 1, j] + img[i + 1, j + 1]) / 255)
#                 if eightField == 7:  # 黑色块1个
#                     endpoint = []
#                     endpoint.append(i)
#                     endpoint.append(j)
#                     endpoints.append(endpoint)
#     for m in range(len(endpoints)):
#         cv2.circle(endpoint2, (endpoints[m][1], endpoints[m][0]), 3, (0, 0, 255), 1)
#
#     show(endpoint2, endpointImg)
#
#
# def bifurcation():
#     global img
#     bifurcation1 = img
#     bifurcation2 = img
#     bifurcations = []
#     h, w = bifurcation1.shape
#     for i in range(1, h - 1):
#         for j in range(1, w - 1):
#
#             if bifurcation1[i, j] == 0:
#
#                 eightField = (bifurcation1[i - 1, j - 1] + bifurcation1[i - 1, j] + bifurcation1[i - 1, j + 1] +
#                               bifurcation1[i, j - 1] + bifurcation1[i, j + 1] + bifurcation1[i + 1, j - 1] +
#                               bifurcation1[i + 1, j] + bifurcation1[i + 1, j + 1]) / 255
#                 if eightField == 5:  # 黑色块3个
#                     bifurcation = []
#                     bifurcation.append(i)
#                     bifurcation.append(j)
#                     bifurcations.append(bifurcation)
#     for m in range(len(bifurcations)):
#         cv2.circle(bifurcation2, (bifurcations[m][1], bifurcations[m][0]), 3, (0, 0, 255), 1)
#
#     show(bifurcation2, bifurcationImg)


# def normalize():
#     global img
#     img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图
#     # 规格化处理
#     h, w = img.shape
#     e = 0
#     for i in range(h):
#         for j in range(w):
#             e += img[i, j]
#     e = e / h / w
#     print(e)
#     v = 0
#     for i in range(h):
#         for j in range(w):
#             v += (img[i, j] - e) ** 2
#     v = v / h / w
#     print(v)
#
#     e0 = 100
#     v0 = 100
#     for i in range(h):
#         for j in range(w):
#             if img[i, j] > e:
#                 img[i, j] = e0 + sqrt((v0 * (img[i, j] - e) ** 2) / v)
#             else:
#                 img[i, j] = e0 - sqrt((v0 * (img[i, j] - e) ** 2) / v)
#     show(img, normImg)


# def directionField():
#     h, w = img.shape
#     hBlockNum = int(h / 16 - 0.5)
#     wBlockNum = int(w / 16 - 0.5)
#
#     directionFieldRow = []
#     directionField = []
#     for i in range(hBlockNum):
#         for j in range(wBlockNum):
#             sobelX = cv2.Sobel(img[16 * i:16 * i + 16, 16 * j:16 * j + 16], cv2.CV_16S, dx=1, dy=0)
#             sobelX = cv2.convertScaleAbs(sobelX)
#             # print(sobelX)
#
#             sobelY = cv2.Sobel(img[16 * i:16 * i + 16, 16 * j:16 * j + 16], cv2.CV_16S, dx=0, dy=1)
#             sobelY = cv2.convertScaleAbs(sobelY)
#
#             temp = 2 * multiply(sobelX, sobelY)
#             Vx = temp.sum()
#
#             temp = sobelX ** 2 - sobelY ** 2
#             Vy = temp.sum()
#
#             theta = 0.5 * atan(Vx / Vy)
#             print(theta)
#             directionFieldRow.append(theta)
#
#         directionField.append(directionFieldRow)
#     directionField = np.array(directionField)
#     directionField = np.tile(directionField, (16, 16))
#     # directionField = cv2.resize(directionField, (16 * wBlockNum, 16 * hBlockNum), interpolation=cv2.INTER_AREA)
#     show(directionField, directionFieldImg)


def show(mImg, label):
    global oriTk, enhanceTk, thinningTk, featureTk

    # h, w = mImg.shape[:2]
    # if h > 200:
    #     w = int(w * 200 / h)
    #     h = 200
    #     mImg = cv2.resize(mImg, (w, h), interpolation=cv2.INTER_AREA)
    mImg = Image.fromarray(mImg)
    if label == oriImg:
        oriTk = ImageTk.PhotoImage(image=mImg)
        label.configure(image=oriTk)
    elif label == enhanceImg:
        enhanceTk = ImageTk.PhotoImage(image=mImg)
        label.configure(image=enhanceTk)
    elif label == thinningImg:
        thinningTk = ImageTk.PhotoImage(image=mImg)
        label.configure(image=thinningTk)
    elif label == featureImg:
        featureTk = ImageTk.PhotoImage(image=mImg)
        label.configure(image=featureTk)
    # elif label == bifurcationImg:
    #     bifurcationTk = ImageTk.PhotoImage(image=mImg)
    #     label.configure(image=bifurcationTk)


if __name__ == '__main__':
    global img, oriTk, enhanceTk, thinningTk, featureTk
    root = Tk()
    root.title("指纹图像预处理和特征提取")
    root.geometry('1050x720')

    btnChoose = Button(root, text="1.选择图片", command=choosePic)
    btnChoose.place(x=50, y=50)

    oriImg = Label(root)
    oriImg.place(x=50, y=100)

    # btnNorm = Button(root, text="规格化", command=normalize)

    btnEnhance = Button(root, text="2.图像增强", command=image_enhance)
    btnEnhance.place(x=350, y=50)

    enhanceImg = Label(root)
    enhanceImg.place(x=350, y=100)

    btnThinning = Button(root, text="3.细化", command=thinning)
    btnThinning.place(x=650, y=50)

    thinningImg = Label(root)
    thinningImg.place(x=650, y=100)

    btnFeature = Button(root, text="4.特征提取及描述", command=feature)
    btnFeature.place(x=50, y=350)

    featureImg = Label(root)
    featureImg.place(x=50, y=400)

    txtFeature = Text(root, height=25, width=95)
    txtFeature.place(x=350, y=350)

    # btnBifurcation = Button(root, text="分叉点特征提取及描述", command=bifurcation)
    # btnBifurcation.place(x=350, y=350)
    #
    # bifurcationImg = Label(root)
    # bifurcationImg.place(x=350, y=400)

    root.mainloop()
