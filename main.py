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
    for v1 in range(6):
        line = f.readline()
        line = line[:-2]
        line = line.split(' ')
        block.append(line)
    data[key] = block
f.close()  # 关闭文件


def similarity(x):
    tidx = 0
    tmax = 0
    for v0 in range(30):
        # idx = rnd.randrange(0, 60, 1)
        if v0 == 0:
            idx = 0
        idx = v0
        tmp = 0
        for v1 in range(6):
            for v2 in range(6):
                if (data[idx][v1][v2] == '1' and x[v1][v2] == 255) or (data[idx][v1][v2] == '0' and x[v1][v2] == 0):
                    # if data[idx][v1][v2] == '1' and x[v1][v2] == 255:
                    tmp = tmp + 1
        if tmp > tmax:
            tmax = tmp
            tidx = idx
        if tmp>25:
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
    img = cv.imread('test2.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.resize(img, (648, 360))
    img = cv.resize(img, (imgWidth, imgHeight))
    imgOri = img
    img = cv.Canny(img, 32, 48)
    idxImg = [[0 for x in range(blockCol)] for x in range(blockRow)]
    subImg = []
    subImgRow = np.split(img, blockRow)
    for v0 in range(blockRow):
        subImg.append(np.split(subImgRow[v0], blockCol, 1))
    for v0 in range(blockRow):
        print(v0 / blockRow)
        for v1 in range(blockCol):
            tmp = similarity(subImg[v0][v1])
            idxImg[v0][v1] = tmp
    for v0 in range(blockRow):
        for v1 in range(blockCol):
            for v2 in range(6):
                for v3 in range(6):
                    if data[idxImg[v0][v1]][v2][v3] == '0':
                        tmp = 0
                    else:
                        tmp = 255
                    img[v0 * 6 + v2][v1 * 6 + v3] = tmp
    _ = 1
    # img=cv.bitwise_or(imgOri,img)
    cv.imshow('result', img)
    cv.waitKey()
    # for v0 in range(15):
    #     for v1 in range(27):
    #         plt.subplot(15, 27, v0 * 27 + v1 + 1)
    #         plt.imshow(subImg[v0][v1])
    #         plt.axis('off')
    # plt.show()
    # cv.imshow('1', subImg[0][0])
    # cv.waitKey()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
