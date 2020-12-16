import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
data = {}
f = open("matrix.txt", "r")  # 设置文件对象
eigBitmap = [0 for x in range(224)]
for v0 in range(224):
    _ = f.readline()
    key = v0
    block = []
    tmp = 0
    pos = 1
    for v1 in range(6):
        line = f.readline()
        line = line[:-2]
        line = line.split(' ')
        for v2 in range(6):
            if line[v2] == '1':
                tmp = tmp + pos
            pos = pos * 2
        block.append(line)
    data[key] = tmp
f.close()  # 关闭文件


def bit_code(x):
    tmp = 0
    pos = 1
    for v1 in range(6):
        for v2 in range(6):
            if x[v1][v2] == 255:
                tmp = tmp + pos
            pos = pos * 2
    return tmp


def similarity(x):
    tidx = 0
    tmax = 0
    for v0 in range(54):
        # idx = rnd.randrange(0, 128, 1)
        idx = v0
        tmp = 0
        sim = x ^ data[idx]
        for v0 in range(36):
            if sim & 1 != 1:
                tmp = tmp + 1
            sim = sim >> 1
        if tmp > tmax:
            tmax = tmp
            tidx = idx
        if tmp > 32:
            return tidx
    return tidx


'''
五个方向
'''
if __name__ == '__main__':
    imgWidth = 648
    imgHeight = 360
    blockRow = int(imgHeight / 6)
    blockCol = int(imgWidth / 6)
    vid = cv.VideoCapture('banana.mp4')
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    vidOut = cv.VideoWriter('banana.avi', fourcc, 30, (648, 360), False)
    frame=0
    while vid.isOpened():
        print(frame)
        frame=frame+1
        ret, img = vid.read()
        # img = cv.imread('test.jpg')
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img = cv.resize(img, (648, 360))
        img = cv.resize(img, (imgWidth, imgHeight))
        imgOri = img
        img = cv.Canny(img, 16, 32)
        idxImg = [[0 for x in range(blockCol)] for x in range(blockRow)]
        subImg = []
        subImgRow = np.split(img, blockRow)
        for v0 in range(blockRow):
            subImg.append(np.split(subImgRow[v0], blockCol, 1))
        for v0 in range(blockRow):
            for v1 in range(blockCol):
                subImg[v0][v1] = bit_code(subImg[v0][v1])
        for v0 in range(blockRow):
            for v1 in range(blockCol):
                tmp = similarity(subImg[v0][v1])
                idxImg[v0][v1] = tmp
        for v0 in range(blockRow):
            for v1 in range(blockCol):
                tmp = data[idxImg[v0][v1]]
                for v2 in range(6):
                    for v3 in range(6):
                        if tmp & 1 == 1:
                            img[v0 * 6 + v2][v1 * 6 + v3] = 255
                        else:
                            img[v0 * 6 + v2][v1 * 6 + v3] = 0
                        tmp = tmp >> 1
        # img = cv.bitwise_or(imgOri, img)
        cv.imshow('result', img)
        vidOut.write(img)
        cv.waitKey(1)
    vid.release()
    vidOut.release()
    # for v0 in range(15):
    #     for v1 in range(27):
    #         plt.subplot(15, 27, v0 * 27 + v1 + 1)
    #         plt.imshow(subImg[v0][v1])
    #         plt.axis('off')
    # plt.show()
    # cv.imshow('1', subImg[0][0])
    # cv.waitKey()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
