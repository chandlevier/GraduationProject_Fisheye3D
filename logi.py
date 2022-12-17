'''
参考http://blog.csdn.net/dengxf01/article/details/53374014
参考http://blog.csdn.net/wd1603926823/article/details/45672741
张伟等 《鱼眼图像校正算法研究》

function [img_valid, R] = imageEffectiveAreaInterception(img, T)
input:
    img: rgb image
    T: gray threshold
output:
    img_valid: effective image area
    R: effective image area radius
'''

import cv2
import numpy as np
import math
import sys

def main(p):
    # rgb to gray
    img = cv2.imread(p)
    img_gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    rows, cols = img_gray.shape[:2]
    print(rows, cols)
    T = 40

    # 从上到下扫描
    for i in range(rows):
        flag = 0
        for j in range(cols):
            if(img_gray[i,j]>=T):
                if(img_gray[i+1,j]>=T):
                    top = i
                    flag = 1
                    break
        if flag == 1:
            break
    print('top =', top)

    # 从下到上扫描
    for i in range(rows-1, top, -1):
        flag = 0
        for j in range(cols):
            if(img_gray[i,j]>=T):
                if(img_gray[i-1,j]>=T):
                    bottom = i
                    flag = 1
                    break
        if flag == 1:
            break
    print('bottom =', bottom)

    # 从左到右扫描
    for j in range(cols):
        flag = 0
        for i in range(top, bottom, 1):
            if(img_gray[i,j]>=T):
                if(img_gray[i+1,j]>=T):
                    left = j
                    flag = 1
                    break
        if flag == 1:
            break
    print('left =', left)

    # 从右到左扫描
    for j in range(cols-1, left, -1):
        flag = 0
        for i in range(top, bottom, 1):
            if(img_gray[i,j]>=T):
                if(img_gray[i-1,j]>=T):
                    right = j
                    flag = 1
                    break
        if flag == 1:
            break
    print('right =', right)

    # 有效区域半径，并提取有效区域
    R = max((bottom-top)/2, (right-left)/2)
    print('R =', R)
    img_valid = img[top:int(top+2*R+1), left:int(left+2*R+1)]
    # cv2.imshow('crop', img_valid)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 经度坐标矫正法：把鱼眼图像中像素的横坐标变换到原来的位置，而纵坐标不变
    # 通过这样的变换会把圆形的鱼眼区域变换成正方形
    x0 = (right-left)/2
    y0 = (bottom-top)/2
    m, n, k = img_valid.shape[:3]
    result = np.zeros((m,n,k))

    for u in range (top, bottom, 1):
        for v in range (left, right, 1):
            i = u
            j = round(math.sqrt(R*R - (u-y0)*(u-y0))*(v-x0)/R + x0)
            if (R*R - (u-y0)*(u-y0)) < 0:
                continue
            result[u, v, 0] = img_valid[i, j, 0]
            result[u, v, 1] = img_valid[i, j, 1]
            result[u, v, 2] = img_valid[i, j, 2]
        # print("外循环", u)

    Undistortion = np.uint8(result)
    a, b, c = Undistortion.shape[:3]
    print('a,b,c',a,b,c)
    Undistortion = cv2.resize(Undistortion, dsize=None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite('undistortion_inter.jpg', Undistortion)

if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)

    