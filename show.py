import numpy as np
np.warnings.filterwarnings("ignore",category =RuntimeWarning)
import cv2
import time
from tqdm import tqdm

NUMBER_OF_FRAMES = 1000

def fisheye_show():
    """
    Loads input images from the path specified and creates a list of
    four-stream graphs generated with the imported images.
    """
    
    print('\nGenerating four-stream graphs...\n')
    
    for i in tqdm(range(NUMBER_OF_FRAMES)):  # Tqdm：快速,可扩展的Python进度条,可以在 Python 长循环中添加一个进度提示信息
    
        image = cv2.imread('./und_bev/frame' + str(i) + '_BEV.png')
        cv2.imshow("1", image)
        cv2.waitKey(50)


def main():

    fisheye_show()
    
    elapsed_time = time.time() - start_time
    print('\nElapsed time: ', elapsed_time ,' seconds')

if __name__ == "__main__":
    start_time = time.time()
    main()

