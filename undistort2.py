'''
使用图像的相机内参K、D、DIM，矫正图像，矫正时终端输入图像，如下：
python3 undistort.py bike_to_undistort.jpg
如何保证图像尺寸不同的图像获得处理
且矫正后的图像边界裁减现象有所改善
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

def undistort(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1] # dim1 is the dimension of input image to un-distort
    # print(dim1) # (576,576)

    assert dim1[0]/dim1[1] == DIM[0]/DIM[1]# 进行矫正的图像和标定用的图像应该具有相同的纵横比，但是尺寸可以不同，尺寸不同时会缩放K的大小
    
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    
    scaled_K = K * dim1[0]/DIM[0] # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0

    # This is how scaled_K, dim2 and balance are used to determine the final K used
    # to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    cv2.imshow("undistorted", undistorted_img)
    cv2.imwrite('bike_outcome2.jpg', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)

