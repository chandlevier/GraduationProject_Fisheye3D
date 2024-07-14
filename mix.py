import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import camera_script

def mix(img1, img2, frame):
    h, w = img1.shape[:2]
    image = np.zeros((h, w, 3), np.uint8)

    image = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    cv2.imwrite("mix/frame" + str(frame) + ".png", image)
    # cv2.imshow("mix", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def main():
    # img1 = cv2.imread("frame1_BEV.png")
    # img2 = cv2.imread("frame000741.png")
    imgs = glob.glob("coop_bev/score2coop4/*.png")
    groundtruths = glob.glob("bev_groundtruth/*.png")
    min_frame = 1000
    for i in groundtruths:
        cur_frame = int(os.path.basename(i)[5:11])
        if cur_frame < min_frame:
            min_frame = cur_frame
    # print(min_frame)
    camera_script.mkdir_folder("/home/piaozx/文档/carla-code/carlafisheye/", "mix", None)
    for i in tqdm(range(len(imgs))):
        img1 = cv2.imread("coop_bev/score2coop3/frame" + str(i) + "_BEV.png")
        # img2 = cv2.imread("merge_bev/vehicle2/frame" + str(i) + "_BEV.png")
        img2 = cv2.imread("bev_groundtruth/frame" + str(min_frame+i).zfill(6) + ".png")
        mix(img1, img2, i)
    # mix(img1, img2, 1)


if __name__ == '__main__':
    main() 
