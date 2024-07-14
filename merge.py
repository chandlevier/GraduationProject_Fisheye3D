# 思路：
# 将后视图水平翻转，右视图顺时针旋转90度，左视图逆时针旋转90度
# 然后将四张图片拓展成1024x1024的同等尺寸
# 再将四张图拍在一张空白图像上
# 空白图像1024x1024
# 具体操作参考tofisheyex4.py或cubemap_script.py

import cv2
from tqdm import tqdm
import numpy as np
import glob
import camera_script
import colorsys
from PIL import Image
from argparse import ArgumentParser
import os

class BBox(object):
    def __init__(self, bbox):
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]

def bev_rotation(img, flag):
    h, w, c = img.shape # 图像的高，宽，通道；注意：OpenCV 先获得高！然后是宽和通道
    box = [0, 0, w, h]  # 创建一个列表，共4项，前两项表示原点，第三项为宽，第四项为高
    bbox = BBox(box)    # 创建BBox对象

    center = (bbox.left + bbox.right)/2, (bbox.top + bbox.bottom)/2 # 计算中心点
    if flag == "fishb":
        rot_mat = cv2.getRotationMatrix2D(center, -180, 1)  # 仿射变换矩阵
    elif flag == "fishr":
        rot_mat = cv2.getRotationMatrix2D(center, -39, 1)
    elif flag == "fishl":
        rot_mat = cv2.getRotationMatrix2D(center, 39, 1)
    img_rot = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), borderValue=(255,255,255))
    # cv2.imwrite("1.png", img_rot)
    return img_rot


def bev_rotate(img, flag):
    if flag == "fishr":
        img_rot = cv2.rotate(img, rotateCode=0)
    elif flag == "fishb":
        img_rot = cv2.rotate(img, rotateCode=1)
    elif flag == "fishl":
        img_rot = cv2.rotate(img, rotateCode=2)
    return img_rot


def bev_norm(image, type):
    output_image = np.zeros((2048, 2048, 3), np.uint8)
    output_image.fill(255)
    h, w = image.shape[0], image.shape[1]
    # 先将图片翻转成正确角度，然后在拓展成1024x1024的格式
    if type == "fishf":
        for row in range(0, h):
            for column in range(0, w):
                output_image[row+512, column+int(w/2)] = image[row, column]
    elif type == "fishb":
        image = bev_rotation(image, type)
        for row in range(0, h):
            for column in range(0, w):
                output_image[row+512-300, column+int(w/2)-20] = image[row, column]
    elif type == "fishr":
        image = bev_rotation(image, type)
        for row in range(0, h):
            for column in range(0, w):
                output_image[row+int(h/2)-30, column+512-100] = image[row, column]
    elif type == "fishl":
        image = bev_rotation(image, type)
        for row in range(0, h):
            for column in range(0, w):
                output_image[row+int(h/2)-40, column+512+100] = image[row, column]
    
    return output_image
        

def mix(img1, img2):
    image = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    # image = np.hstack((img1, img2))
    return image


def merge_bevs(input_file):
    """
    Loads input images from the path specified and creates a list of
    cubemaps generated with the imported images.
    :param pointer: indicates the position of the window
    :return: A list of cubemaps.
    """
    
    print('\nGenerating total bird\'s eye view...\n')

    # output_image = np.zeros((724,1002,3))
    nums = glob.glob(input_file+"/fishf"+"/*.png")
    for i in range(len(nums)):
        frame = "frame"+str(i)
        #-------------------- 读取各个视图鸟瞰图 --------------------#
        front = cv2.imread(input_file + '/fishf/' + frame + '_bev.png')
        back = cv2.imread(input_file + '/fishb/' + frame + '_bev.png')
        right = cv2.imread(input_file + '/fishr/' + frame + '_bev.png')
        left = cv2.imread(input_file + '/fishl/' + frame + '_bev.png')
        #-------------------- 处理后、左、右视图 --------------------#
        front = bev_norm(front, "fishf")
        back = bev_norm(back, "fishb")
        right = bev_norm(right, "fishr")
        left = bev_norm(left, "fishl")

        img1 = mix(front, back) 
        img2 = mix(right, left)
        output_image = mix(img1, img2)
        
        # cv2.imshow("1", output_image)
        # cv2.waitKey(0)
        camera_script.mkdir_folder("/home/piaozx/文档/carla-code/carlafisheye", input_file+"_total", None)
        fname = input_file + '_total/' + frame + '_BEV.png'
        cv2.imwrite(fname, output_image)

        
def groundtruth_norm():
    camera_script.mkdir_folder("/home/piaozx/文档/carla-code/carlafisheye", "bev_gtnorm", None)
    bev_groundtruth = glob.glob("bev_groundtruth/*.png")
    for fname in tqdm(bev_groundtruth):
        bev_gt = cv2.imread(fname)
        bev_gtnorm = np.zeros((1024, 1024, 3), np.uint8)
        bev_gtnorm.fill(255)
        for r in range(0, 512):
            for c in range(0, 512):
                bev_gtnorm[r+256, c+256] = bev_gt[r, c]
        filename = "bev_gtnorm" + os.path.basename(fname)
        cv2.imwrite(filename, bev_gtnorm)



def main():
    """
    main function
    """
    input_file = ["und_bev", "dis_bev"]
    # parser = ArgumentParser()
    # parser.add_argument('input_file', help='input files')
    # args = parser.parse_args()
    for i in tqdm(range(len(input_file))):
        merge_bevs(input_file[i])

    # 将鸟瞰真值也整理成1024x1024的大小
    # groundtruth_norm()


if __name__ == "__main__":
    main()




