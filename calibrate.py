'''
获取图像的相机内参K、D、DIM,最终可获得结果如下:
Found 1 valid images for calibration
DIM=(1344, 1344)
K=np.array([[495.9548026014525, 0.0, 661.5037797849802], [0.0, 495.3069256164778, 650.0623212604137], [0.0, 0.0, 1.0]])
D=np.array([[-0.05828770288339518], [-0.06981654150537402], [0.1208910189860674], [-0.05683034290476975]])
'''

import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (6, 9)

# 找棋盘格角点(角点精准化迭代过程的终止条件)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# 不同的标志，可能是0或以下值的组合:
# fisheye::CALIB_RECOMPUTE_EXTRINSIC 每次迭代优化内参后，将重新计算外参
# fisheye::CALIB_CHECK_COND 函数将检查条件数的有效性
# fisheye::CALIB_FIX_SKEW 倾斜系数(alpha)设置为零,并保持为零
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg') # 大概是获取图像存储路径文件夹中所有以".jpg"为结尾的文件路径
for fname in images:        # 遍历所有jpg图像，确保所有图像的尺寸相同，否则给出警告
    img = cv2.imread(fname)
    if _img_shape == None:      # 如果是第一次输入图像，_img_shape还没有具体值，那么将这批图像的尺寸赋给它
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 将尺寸正确的输入图像转化为灰度图

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
    cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)
        imgpoints.append(corners)
# 符合要求的图像数量
N_OK = len(objpoints)
# 初始化内参矩阵K,畸变系数D,旋转向量rvecs,平移向量tvecs
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
cv2.fisheye.calibrate(
                        objpoints,
                        imgpoints,
                        gray.shape[::-1],
                        K,
                        D,
                        rvecs,
                        tvecs,
                        calibration_flags,
                        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

