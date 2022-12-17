class Config(object):
    pi = 3.1415926535
    n = 10
    # 图片大小，初始化为0，后续在main函数中会读取输入图像的长宽
    l = 0
    w = 0
    # 灰度阈值，用于切点判断
    thre = 50
    # 路径
    path = '/home/piaozx/文档/opencv/linescanning/bike_500.jpg'
    res_path = '/home/piaozx/文档/opencv/linescanning/bike_undistort.jpg'
    omiga = 0
