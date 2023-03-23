import json
import glob
import os
import cv2
from argparse import ArgumentParser
import camera_script
from tqdm import tqdm

# img_path = "/home/piaozx/文档/carla-code/cube2fisheye/output/"
# 最终存储路径:"json/fishf/framexxx.png"

def create_json(img_path, frame, camera, h, w, f):
    control = {"images":[
                        {
                        "file_name": img_path + "/" + frame + ".png", 
                        "cam_intrinsic": [
                                [f, 0.0, w/2],
                                [0.0, f, h/2],
                                [0.0, 0.0, 1.0]],
                        "width": w,
                        "height": h
                        }]}
    camera_script.mkdir_folder("json/", camera, None)
    json.dump(control, open('./json/'+camera + "/" + frame+'.json','w'), indent=4)# indent 缩进

def main():
    # image_file ./output/fishf或./output/fishb或./output/fishr或./output/fishl
    # os.path.basename(image_file) = "fishf"或"fishb"...
    # parser = ArgumentParser()
    # parser.add_argument('image_file', help='image files')
    # args = parser.parse_args()
    
    images_file = ['./output/fishf', './output/fishb', './output/fishr', './output/fishl']
    for i in tqdm(range(4)):
        image_file = images_file[i]
        images = glob.glob(image_file + '/*.png')
        focus = 1000       # 相机焦距
        for fname in images:
            # print(os.path.basename(fname))
            img = cv2.imread(fname)
            h = img.shape[0]
            w = img.shape[1]
            camera = os.path.basename(image_file)
            frame = os.path.splitext(os.path.basename(fname))[0] # e.g. frame = frame67
            create_json(image_file, frame, camera, h, w, focus)

if __name__ == "__main__":
    main()
