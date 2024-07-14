import json
import glob
import os
import cv2
import camera_script
from tqdm import tqdm

# img_path = "/home/piaozx/文档/carla-code/carlafisheye/output/"
# 最终存储路径:"json/fishf/framexxx.png"
# vehicle = "json1/"

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
    json.dump(control, open('./json/'+ camera + "/" + frame+'.json','w'), indent=4)# indent 缩进


def create_multijson(img_path, frame, vehicle, camera, h, w, f):
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
    camera_script.mkdir_folder('./dataset/'+ vehicle, camera, None)
    json.dump(control, open('./dataset/'+ vehicle + camera + "/" + frame+'.json','w'), indent=4)# indent 缩进


def json_script(n, flag):   
    if flag == 1:
        images_file = ['./output/fishf', './output/fishb', './output/fishr', './output/fishl']
        for i in tqdm(range(len(images_file))):
            image_file = images_file[i]
            frame = "/frame" + str(n) + ".png"
            fname = image_file + frame
            # fname = glob.glob(image_file + frame)[0]
            focus = 1000       # 相机焦距

            # print(os.path.basename(fname))
            img = cv2.imread(fname)
            h, w = img.shape[:2]
            camera = os.path.basename(image_file)   # "fishf"
            frame = os.path.splitext(os.path.basename(fname))[0] # e.g. frame = frame67
            create_json(image_file, frame, camera, h, w, focus)
    elif flag == 2:
        images_file = ['./output1/fishf', './output1/fishb', './output1/fishr', './output1/fishl', './output2/fishf', './output2/fishb', './output2/fishr', './output2/fishl']
        for i in tqdm(range(len(images_file))):
            image_file = images_file[i]
            fname = image_file + "/frame" + str(n) + ".png"
            # fname = glob.glob(image_file + frame)[0]
            focus = 1000       # 相机焦距
            img = cv2.imread(fname)
            h, w = img.shape[:2]
            camera = os.path.basename(image_file)   # "fishf"
            frame = os.path.splitext(os.path.basename(fname))[0] # e.g. frame = frame67
    
            if i>=0 and i < 4:
                create_multijson(image_file, frame, "json1/", camera, h, w, focus)
            elif i>=4 and i<8:
                create_multijson(image_file, frame, "json2/", camera, h, w, focus)
    elif flag == 3:
        images_file = ['./dataset/output1/fishf', './dataset/output1/fishb', './dataset/output1/fishr', './dataset/output1/fishl', './dataset/output2/fishf', './dataset/output2/fishb', './dataset/output2/fishr', './dataset/output2/fishl', './dataset/output3/fishf', './dataset/output3/fishb', './dataset/output3/fishr', './dataset/output3/fishl']
        for i in tqdm(range(len(images_file))):
            image_file = images_file[i]
            fname = image_file + "/frame" + str(n) + ".png"
            # fname = glob.glob(image_file + frame)[0]
            focus = 1000       # 相机焦距
            img = cv2.imread(fname)
            h, w = img.shape[:2]
            camera = os.path.basename(image_file)   # "fishf"
            frame = os.path.splitext(os.path.basename(fname))[0] # e.g. frame = frame67
    
            if i>=0 and i < 4:
                create_multijson(image_file, frame, "json1/", camera, h, w, focus)
            elif i>=4 and i<8:
                create_multijson(image_file, frame, "json2/", camera, h, w, focus)
            elif i>=8 and i<12:
                create_multijson(image_file, frame, "json3/", camera, h, w, focus)
