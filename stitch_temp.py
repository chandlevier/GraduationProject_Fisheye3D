import numpy as np

np.warnings.filterwarnings("ignore",category =RuntimeWarning)

import cv2

import time

from tqdm import tqdm

import glob
import camera_script

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

    output_image = np.zeros((2688,2688,3))

    front = cv2.imread('./und_pred/fishf/frame' + str(26) + '_pred.png')
    right = cv2.imread('./und_pred/fishr/frame' + str(26) + '_pred.png')
    back = cv2.imread('./und_pred/fishb/frame' + str(26) + '_pred.png')
    left = cv2.imread('./und_pred/fishl/frame' + str(26) + '_pred.png')

    # 将4张鱼眼图片拼接成四流图的样子，并保存至output_image变量中
    if not front is None:
    
        h = front.shape[0] # 1344
        w = front.shape[1] # 1344

        output_image[0:h, 0:w] = front
        output_image[0:h, w:w+w] = back
        output_image[h:h+h, 0:w] = left
        output_image[h:h+h, w:w+w] = right

        cv2.imwrite('frame' + str(26) + '_und_pred.png', output_image)

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

