# -*- coding: utf-8 -*-

"""

This module transforms six input images that form a cube map into a
fisheye image.

Folders tree is the following:

    Carla_cubemap2fisheye
        |
        |--> main_script.py
        |--> cube2fisheye.py
        |--> output
        |--> input
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

#-------------------------   Constants to define   --------------------#

NUMBER_OF_FRAMES = 10000
WINDOW_SIZE = 4
FACE_SIZE = 1024
FOV = 196 # degrees
output_width = 1344
output_height = 1344
save_path = '/home/piaozx/文档/carla-code/cube2fisheye/'

#----------------------------------------------------------------------#

def loadImages(pointer):
    """
    Loads input images from the path specified and creates a list of
    cubemaps generated with the imported images.
    :param pointer: indicates the position of the window
    :return: A list of cubemaps.
    """
    
    print('\nGenerating cubemaps...\n')
    
    cubemap = []
    
    for i in tqdm(range(WINDOW_SIZE)):  # Tqdm：快速,可扩展的Python进度条,可以在 Python 长循环中添加一个进度提示信息
    
        output_image = np.zeros((3072,4096,3))
    
        if i + pointer < 10:
            zero_string = 5*'0'
        elif i + pointer >= 10 and i + pointer < 100:
            zero_string = 4*'0'
        elif i + pointer >= 100 and i + pointer < 1000:
            zero_string = 3*'0'
        elif i + pointer >= 1000 and i + pointer < 10000:
            zero_string = 2*'0'
        elif i + pointer >= 10000 and i + pointer < 100000:
            zero_string = 1*'0'
        elif i + pointer >= 100000 and i + pointer < 1000000:
            zero_string = 0*'0'
        #-------------------- 每个鱼眼相机修改一次 --------------------#
        front = cv2.imread('./input/fishb/front/' + zero_string +
                            str(pointer + i) +
                            '.png')
        left = cv2.imread('./input/fishb/left/' + zero_string +
                            str(pointer + i) +
                            '.png')
        right = cv2.imread('./input/fishb/right/' + zero_string +
                            str(pointer + i) +
                            '.png')
        top = cv2.imread('./input/fishb/top/' + zero_string +
                            str(pointer + i) +
                            '.png')
        bottom = cv2.imread('./input/fishb/bottom/' + zero_string +
                            str(pointer + i) +
                            '.png')
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
            
            cv2.imwrite('./cubemaps/frame' + str(i) + '.png', output_image)
            
            cubemap.append(output_image)

    return cubemap

def main():
    """
    main function
    """
    
    cubemap_to_fisheye()
    
    elapsed_time = time.time() - start_time
    
    print('\nElapsed time: ', elapsed_time ,' seconds')

def cubemap_to_fisheye():
   
    """
    Converts loaded cube maps into fisheye images
    """
    
    # Create new output image with the dimentions computed above
    output_image = np.zeros((output_height,output_width,3))
    fov = FOV*np.pi/180
    
    # counter allows for correct naming when cropping
    pointer = 0
    # 确定输出图像平面上每个点的极坐标表达式，返回数组r, phi
    r, phi = get_spherical_coordinates(output_height, output_width)
    # 将输出图像上的二维极坐标转化为三维空间中对应的空间点坐标
    x, y, z = spherical_to_cartesian(r, phi, fov)
    
    number_of_frames = 0
    
    while number_of_frames < NUMBER_OF_FRAMES:
        cubemap = loadImages(pointer)
        print('\nCubemaps frame%d successfully loaded...\n' % number_of_frames)
        for image in tqdm(cubemap):
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
            #-------------------- 每个鱼眼相机修改一次 --------------------#
            camera_script.mkdir_folder(save_path, 'output', 'fishb')
            # 将鱼眼图像关于y轴翻转一下子
            output_image = cv2.flip(output_image, 1)
            cv2.imwrite('./output/fishb/frame' +
                        str(number_of_frames) +
                        '.png', output_image)
            #-------------------- 每个鱼眼相机修改一次 --------------------#
            number_of_frames += 1
        pointer += WINDOW_SIZE  # 为啥一下要加5？
    return

def cubic_interpolation(image, x_cubemap, y_cubemap):
    """
    Performs a cubic interpolation of a given pixel
    :param image: image where the pixel to interpolate belongs
    :param x_cubemap: x pixel coordinate on the cube map
    :param y_cubemap: y pixel coordinate on the cube map
    :return: resulting RGB values for the interpolated pixel
    """
    try:
        top_pixel = (y_cubemap + 1, x_cubemap)
    except IndexError:
        top_pixel = (y_cubemap, x_cubemap)
    
    try:
        right_pixel = (y_cubemap, x_cubemap + 1)
    except IndexError:
        right_pixel = (y_cubemap, x_cubemap)
    
    try:
        bottom_pixel = (y_cubemap - 1, x_cubemap)
    except IndexError:
        bottom_pixel = (y_cubemap, x_cubemap)
    
    try:
        left_pixel = (y_cubemap, x_cubemap - 1)
    except IndexError:
        left_pixel = (y_cubemap, x_cubemap)
    
    mean_red = (image[top_pixel[0], top_pixel[1], 2] +
                image[right_pixel[0], right_pixel[1], 2] +
                image[left_pixel[0], left_pixel[1], 2] +
                image[bottom_pixel[0], bottom_pixel[1], 2])/4
    x_inter_R = 0*image[y_cubemap, x_cubemap, 2] + 1*mean_red

    mean_green = (image[top_pixel[0], top_pixel[1], 1] +
                    image[right_pixel[0], right_pixel[1], 1] +
                    image[left_pixel[0], left_pixel[1], 1] +
                    image[bottom_pixel[0], bottom_pixel[1], 1])/4
    x_inter_G = 0*image[y_cubemap, x_cubemap, 1] +1*mean_green

    mean_blue = (image[top_pixel[0], top_pixel[1], 0] +
                image[right_pixel[0], right_pixel[1], 0] +
                image[left_pixel[0], left_pixel[1], 0] +
                image[bottom_pixel[0], bottom_pixel[1], 0])/4
    x_inter_B = 0*image[y_cubemap, x_cubemap, 0] + 1*mean_blue
    
    return x_inter_R, x_inter_G, x_inter_B

if __name__ == "__main__":
    start_time = time.time()
    main()

