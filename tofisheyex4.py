# -*- coding: utf-8 -*-

"""

This module transforms six input images that form a cube map into a
fisheye image.

Folders tree is the following:

    Carla_cubemap2fisheye
        |
        |--> main_script.py
        |--> cube2fisheye.py
        |--> output1
        |--> output2
        |--> vehicle1
        |--> vehicle2
            |
            |--> fishb
                |
                |--> front
                |--> right
                |--> left
                |--> top
                |--> bottom
            |--> fishf
                |
                |--> front
                |--> right
                |--> left
                |--> top
                |--> bottom
            |--> fishl
                |
                |--> front
                |--> right
                |--> left
                |--> top
                |--> bottom
            |--> fishr
                |
                |--> front
                |--> right
                |--> left
                |--> top
                |--> bottom
            

Input images must be stored at folders front, right, left, back, top and 
bottom.
Name formatting is of the type:
    '0000', '0001', ... , '0010', '0011', ... , '0100', '0101',etc


Example:
    Type the following command on an opened terminal in order to run the
script
    
    $ python main_script.py

"""

import numpy as np

np.warnings.filterwarnings("ignore",category =RuntimeWarning)

import cv2

import time

from tqdm import tqdm

from cube2fisheye import *

import camera_script

from argparse import ArgumentParser

import glob

#-------------------------   Constants to define   --------------------#

save_path = '/home/piaozx/文档/carla-code/carlafisheye/dataset/'
# images = glob.glob(save_path + "input/fishf/front/*.png")
# NUMBER_OF_FRAMES = 1000
# print(len(images))
WINDOW_SIZE = 4
FACE_SIZE = 1024
FOV = 196 # degrees
output_width = 1344
output_height = 1344

#----------------------------------------------------------------------#

def loadImages(input_file, frame, n):
    """
    Loads input images from the path specified and creates a list of
    cubemaps generated with the imported images.
    :param pointer: indicates the position of the window
    :return: A list of cubemaps.
    """
    
    print('\nGenerating cubemaps...\n')
   
    output_image = np.zeros((3072,3072,3))

    
    #-------------------- 每个鱼眼相机修改一次 --------------------#
    front = cv2.imread(input_file + '/front/' + frame + '.png')
    left = cv2.imread(input_file + '/left/' + frame + '.png')
    right = cv2.imread(input_file + '/right/' + frame + '.png')
    top = cv2.imread(input_file + '/top/' + frame + '.png')
    bottom = cv2.imread(input_file + '/bottom/' + frame + '.png')
    #-------------------- 每个鱼眼相机修改一次 --------------------#
    # 将5张图片拼接成立方体贴图的样子，并保存至output_image变量中，加入cubemap数组中
    if not front is None:
    
        h = front.shape[0] # 1024
        w = front.shape[1] # 1024
    
        output_image[h:h+h, 0:w] = left
        output_image[h:h+h, w:w+w] = front
        output_image[h:h+h, 2*w:2*w+w] = right
        output_image[0:h, w:w+w] = top
        output_image[2*h:2*h+h, w:w+w] = bottom
        
        cv2.imwrite('./cubemaps/frame' + str(n) + '.png', output_image)

    return output_image

def transform(input_file, frame, n, cam):   # frame为str类型
    """
    main function
    """
    # input_file ./vehicle1/fishf ./vehicle1/fishb ./vehicle1/fishr ./vehicle1/fishl    
    cubemap_to_fisheye(input_file, frame, n, cam)
    


def cubemap_to_fisheye(input_file, frame, n, cam):
   
    """
    Converts loaded cube maps into fisheye images
    """
    
    # Create new output image with the dimentions computed above
    output_image = np.zeros((output_height,output_width,3))
    fov = FOV*np.pi/180
    
    # 确定输出图像平面上每个点的极坐标表达式，返回数组r, phi
    r, phi = get_spherical_coordinates(output_height, output_width)
    # 将输出图像上的二维极坐标转化为三维空间中对应的空间点坐标
    x, y, z = spherical_to_cartesian(r, phi, fov) 

    cubemap = loadImages(input_file, frame, n)
    print('\nCubemaps frame%d_%s successfully loaded...\n' % (n, cam))
    image = cubemap
    #-------------------- 将一帧cubemap转换成鱼眼 --------------------#
    for row in range(0, output_height):
        for column in range(0, output_width):
            if np.isnan(r[row, column]):    # 将输出图像平面上极坐标距离超过1的点设置为黑色
                
                output_image[row, column, 0] = 0
                output_image[row, column, 1] = 0
                output_image[row, column, 2] = 0
            # 对于极坐标距离在1以内的点
            else:
                # 首先确定该点对应的三维坐标指向哪个立方体表面
                face = get_face(x[row, column],
                                y[row, column],
                                z[row, column])
                # 然后确定该点在该立方体表面上的uv坐标
                u, v = raw_face_coordinates(face,
                                            x[row, column],
                                            y[row, column],
                                            z[row, column])
                # 最后获取标准化的uv坐标，锁定输出图像上坐标为(row,column)的点对应着立方体哪个表面上的哪个点
                xn, yn = normalized_coordinates(face,
                                                u,
                                                v,
                                                FACE_SIZE)
                # 将原图像素色彩值转移到输出图像上
                output_image[row, column, 0] = image[yn, xn, 0]
                output_image[row, column, 1] = image[yn, xn, 1]
                output_image[row, column, 2] = image[yn, xn, 2]
    #-------------------- 转换完一帧图像后进行存储 --------------------#
    # 将鱼眼图像关于y轴翻转一下子
    output_image = cv2.flip(output_image, 1)
    # input_file[-7] = 1或2
    if input_file[-1] == 'f':
        camera_script.mkdir_folder(save_path, 'output'+str(input_file[-7]), 'fishf')
        cv2.imwrite('./dataset/output'+str(input_file[-7])+'/fishf/frame' +
                    str(n) + '.png', output_image)
    elif input_file[-1] == 'b':
        camera_script.mkdir_folder(save_path, 'output'+str(input_file[-7]), 'fishb')
        cv2.imwrite('./dataset/output'+str(input_file[-7])+'/fishb/frame' +
                    str(n) + '.png', output_image)
    elif input_file[-1] == 'r':
        camera_script.mkdir_folder(save_path, 'output'+str(input_file[-7]), 'fishr')
        cv2.imwrite('./dataset/output'+str(input_file[-7])+'/fishr/frame' +
                    str(n) + '.png', output_image)
    elif input_file[-1] == 'l':
        camera_script.mkdir_folder(save_path, 'output'+str(input_file[-7]), 'fishl')
        cv2.imwrite('./dataset/output'+str(input_file[-7])+'/fishl/frame' +
                    str(n) + '.png', output_image)

    # if input_file[-1] == 'f':
    #     camera_script.mkdir_folder(save_path, 'output', 'fishf')
    #     cv2.imwrite('./output'+'/fishf/frame' +
    #                 str(n) + '.png', output_image)
    # elif input_file[-1] == 'b':
    #     camera_script.mkdir_folder(save_path, 'output', 'fishb')
    #     cv2.imwrite('./output'+'/fishb/frame' +
    #                 str(n) + '.png', output_image)
    # elif input_file[-1] == 'r':
    #     camera_script.mkdir_folder(save_path, 'output', 'fishr')
    #     cv2.imwrite('./output'+'/fishr/frame' +
    #                 str(n) + '.png', output_image)
    # elif input_file[-1] == 'l':
    #     camera_script.mkdir_folder(save_path, 'output', 'fishl')
    #     cv2.imwrite('./output'+'/fishl/frame' +
    #                 str(n) + '.png', output_image)

    return


