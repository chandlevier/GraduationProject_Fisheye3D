# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import sys

def inter(img, height, width):
    src_size = [height, width]
    h, w, c = img.shape
    src = np.zeros((src_size[0], src_size[1], 3), dtype=np.uint8)
    if h == src_size[0] and w == src_size[1]:
        return img
    for i in range(src_size[0]):
        for j in range(src_size[1]):
            # round()四舍五入的函数
            src_x = round(i * (h / src_size[0]))
            src_y = round(j * (w / src_size[1]))
            src[i, j] = img[src_x, src_y]
    return src


def main(p):
    # 读取鱼眼图片，转换为灰度图片
    img = cv2.imread(p)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 设置灰度阈值，并提取原图大小
    T = 40
    rows, cols = img.shape[:2]
    print(rows, cols)

    # 从上向下扫描
    for i in range(0, rows-1, 1):
        for j in range(0, cols, 1):
            if img_gray[i, j] >= T:
                if img_gray[i+1, j] >= T:
                    bottom = i
                    break
            else:
                continue
            break
    print('bottom =', bottom)

    # 从下向上扫描
    for i in range(rows-1, -1, -1):
        for j in range(0, cols, 1):
            if img_gray[i, j] >= T:
                if img_gray[i-1, j] >= T:
                    top = i
                    break
            else:
                continue
            break
    print('top =', top)


    # 从左向右扫描
    for j in range(0, cols-1, 1):
        for i in range(0, rows, 1):
            if img_gray[i, j] >= T:
                if img_gray[i, j+1] >= T:
                    right = j
                    break
            else:
                continue
            break
    print('right =', right)

    # 从右向左扫描
    for j in range(cols-1, -1, -1):
        for i in range(0, rows, 1):
            if img_gray[i, j] >= T:
                if img_gray[i, j-1] >= T:
                    left = j
                    break
            else:
                continue
            break
    print('left =', left)


    '''
    bike_500.jpg扫描结果：
    top = 485, bottom = 72, left = 490, right = 51
    '''

    # 计算有效区域半径
    R = max((bottom-top)/2, (right-left)/2)
    print('R =', R)

    # 提取有效区域
    img_valid = img[top:int(top+2*R+1), left:int(left+2*R+1)]
    # img_valid = img
    cv2.imwrite('result.jpg', img_valid)

    # 经度矫正法(开始看不懂了)
    m, n, k = img_valid.shape[:3]
    print('m, n, k', m, n, k) # 高m=428 宽n=440 色彩k=3
    result = np.zeros((m,n,k))

    for i in range(m): # 从上到下
        for j in range(n): # 从左到右
            u = j - R      # 先转换出来-1到1之间的规格化坐标(u,v)
            v = R - i
            r = math.sqrt(u*u + v*v) # 极坐标系(r,fi)
            if(r == 0):
                fi = 0
            elif(u >= 0):  # 第一象限，fi=arcsin(v/r)
                fi = math.asin(v/r)
            else:          # 第二象限，fi=pi-arcsin(v/r)
                fi = math.pi - math.asin(v/r)
            # r可以映射成球坐标系的θ，θ=r*α/2，α是鱼眼视场角FoV，这里默认为pi
            # fi直接用于从照相机到实景方向向量的球坐标系的ф
            # f为球面半径
            f = R * 2 / math.pi
            theta = r / f
            # f = 1
            # theta = r * math.pi /2
            # f = R * 2 / math.pi
            # theta = r/f
            # theta = (r / R) * math.pi / 2

            x = f * math.sin(theta) * math.cos(fi)
            y = f * math.sin(theta) * math.sin(fi)
            z = f * math.cos(theta)
            # 新选取的球面坐标系的坐标
            rr= math.sqrt(x * x + z * z)
            sita = math.pi / 2 - math.atan( y /rr)
            if(z>=0):
                fai = math.acos(x/rr)
            else:
                fai= math.pi - math.acos(x/rr)
            # round()四舍五入的函数
            # xx = round(f * sita) # 应用这个式子会导致结果图像镜像翻转
            # yy = round(f * fai)
            xx = round(f * sita)
            yy = round(2 * R - f * fai)

            if ((xx < 1) | (yy < 1) | (xx > m) | (yy > n)):
                continue

            result[xx,yy,0] = img_valid[i-1, j-1, 0]
            result[xx,yy,1] = img_valid[i-1, j-1, 1]
            result[xx,yy,2] = img_valid[i-1, j-1, 2]
            # print("内循环",j)
        print("外循环", i)

    Undistortion = np.uint8(result)
    a, b, c = Undistortion.shape[:3]
    print('a,b,c',a,b,c)
    # 插值处理
    # Undistortion = inter(Unditortion, a, b)
    Undistortion = cv2.resize(Undistortion, dsize=None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite('undistortion_inter.jpg', Undistortion)  

    # 显示图片
    cv2.namedWindow("original", 0)
    cv2.resizeWindow("original", 640, 480)
    cv2.imshow("original", img)

    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 640, 480)
    cv2.imshow("result", Undistortion)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)

