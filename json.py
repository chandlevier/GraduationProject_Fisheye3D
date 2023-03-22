import json
import glob
import os
import cv2

img_path = "/home/piaozx/文档/carla-code/cube2test/output/"

def create_json(img_path, name, h, w, f):
    control = {"images":[
                        {
                        "file_name": img_path + name + ".png", 
                        "cam_intrinsic": [
                                [f, 0.0, w/2],
                                [0.0, f, h/2],
                                [0.0, 0.0, 1.0]],
                        "width": w,
                        "height": h
                        }]}

    json.dump(control, open('json/'+name+'.json','w'), indent=4)# indent 缩进

def main():
    images = glob.glob('output/*.png')
    focus = 1000       # 相机焦距
    for fname in images:
        # print(os.path.basename(fname))
        img = cv2.imread(fname)
        h = img.shape[0]
        w = img.shape[1]
        imgname = os.path.splitext(os.path.basename(fname))[0] 
        create_json(img_path, imgname, h, w, focus)

if __name__ == "__main__":
    main()
