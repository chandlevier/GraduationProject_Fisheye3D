'''
使用图像的相机内参K、D、DIM,矫正图像,矫正时终端输入图像,如下:
python3 undistort.py bike_to_undistort.jpg
要求图像尺寸要和获取参数用的图像尺寸相同
矫正后的图像边界会有所裁减
否则参考tutorial2
'''

import cv2
import numpy as np
import os
import glob
import sys
from tqdm import tqdm
import camera_script

DIM=(1344, 1344)
K=np.array([[495.9548026014525, 0.0, 661.5037797849802], [0.0, 495.3069256164778, 650.0623212604137], [0.0, 0.0, 1.0]])
D=np.array([[-0.05828770288339518], [-0.06981654150537402], [0.1208910189860674], [-0.05683034290476975]])


def undistort(img, cam, img_name):
    # img = cv2.imread(img_path)
    h, w = img.shape[:2]

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    camera_script.mkdir_folder('/home/piaozx/文档/carla-code/carlafisheye/und/', cam, None)
    cv2.imwrite('/home/piaozx/文档/carla-code/carlafisheye/und/' + cam + img_name + '.png', undistorted_img)


def undistort_script(n):
    camera = ["fishf/", "fishb/", "fishr/", "fishl/"]
    for i in range(len(camera)):
        cam = camera[i]
        frame = "frame" + str(n)
        fname = "output/" + cam + frame + ".png"
        image = cv2.imread(fname)
        undistort(image, cam, frame)

