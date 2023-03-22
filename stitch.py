# -*- coding: utf-8 -*-

"""

将一辆车上四个鱼眼相机获取的图像按照
    前 后
    左 右
的顺序拼接成一个四流图，并存入单独的文件夹中

Folders tree is the following:

    Carla_cubemap2fisheye
        |
        |--> stitch.py
        |--> cube2fisheye.py
        |--> output
            |
            |--> fishb
            |--> fishf
            |--> fishl
            |--> fishr
        |--> outcome
        |--> input
Input images must be stored at fishb, fishf, fishl, fishr.
Name formatting is of the type:
    'frame0.png', 'frame1.png', ... , 'frame100.png', 'frame101.png',etc


Example:
    Type the following command on an opened terminal in order to run the
script
    
    $ python3 stitch.py

"""

import numpy as np

np.warnings.filterwarnings("ignore",category =RuntimeWarning)

import cv2

import time

from tqdm import tqdm

#-------------------------   Constants to define   --------------------#

NUMBER_OF_FRAMES = 1000
output_width = 1344
output_height = 1344

#----------------------------------------------------------------------#

def fisheye_stitch():
    """
    Loads input images from the path specified and creates a list of
    four-stream graphs generated with the imported images.
    """
    
    print('\nGenerating four-stream graphs...\n')

    
    for i in tqdm(range(NUMBER_OF_FRAMES)):  # Tqdm：快速,可扩展的Python进度条,可以在 Python 长循环中添加一个进度提示信息
    
        output_image = np.zeros((2688,2688,3))

        front = cv2.imread('./output/fishf/frame' + str(i) + '.png')
        right = cv2.imread('./output/fishr/frame' + str(i) + '.png')
        back = cv2.imread('./output/fishb/frame' + str(i) + '.png')
        left = cv2.imread('./output/fishl/frame' + str(i) + '.png')

        # 将4张鱼眼图片拼接成四流图的样子，并保存至output_image变量中
        if not front is None:
        
            h = front.shape[0] # 1344
            w = front.shape[1] # 1344

            output_image[0:h, 0:w] = front
            output_image[0:h, w:w+w] = back
            output_image[h:h+h, 0:w] = left
            output_image[h:h+h, w:w+w] = right

            cv2.imwrite('./outcome/frame' + str(i) + '.png', output_image)

def main():
    """
    main function
    """
    fisheye_stitch()
    
    elapsed_time = time.time() - start_time
    
    print('\nElapsed time: ', elapsed_time ,' seconds')

if __name__ == "__main__":
    start_time = time.time()
    main()

