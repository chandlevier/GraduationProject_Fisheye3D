'''
使用图像的相机内参K、D、DIM，矫正图像，矫正时终端输入图像，如下：
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

# DIM=(768, 576)
# K=np.array([[181.46371348236477, 0.0, 378.44944181081524], [0.0, 181.08546448685613, 278.8436498386038], [0.0, 0.0, 1.0]])
# D=np.array([[0.041429145698135876], [-0.06385380015855485], [0.08243073627114914], [-0.036282249401211684]])
DIM=(576, 576)
K=np.array([[181.96088967058222, 0.0, 283.46072680337977], [0.0, 181.58514011922955, 278.8570174810165], [0.0, 0.0, 1.0]])
D=np.array([[0.039972148252636], [-0.06332675794255915], [0.08153259625893983], [-0.035873064837040815]])

def undistort(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.imwrite('outcome_'+img_path, undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)

