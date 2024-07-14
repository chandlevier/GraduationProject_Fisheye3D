'''
终端运行程序时需要输入图像文件和标注文件的绝对路径

'''
from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)
import glob
import os
from tqdm import tqdm
import copy

# fcos3d模型 score_thr=0.2
config_file = '/home/piaozx/mmdetection3d-conda/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py'
checkpoints_file = '/home/piaozx/mmdetection3d-conda/checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'


def multi_detection3d(n, ego_loc1, ego_loc2, ego_loc3):
    boundingboxes1 = []
    boundingboxes2 = []
    boundingboxes3 = []
    print(id(boundingboxes1), id(boundingboxes2), id(boundingboxes3))
    model = init_model(config_file, checkpoints_file, device='cuda:0')
    # 额外设置变量方便调试用
    image1_file = '/home/piaozx/文档/carla-code/carlafisheye/output1/'
    image2_file = '/home/piaozx/文档/carla-code/carlafisheye/output2/'
    image3_file = '/home/piaozx/文档/carla-code/carlafisheye/output3/'
    ann1_file = '/home/piaozx/文档/carla-code/carlafisheye/json1/'
    ann2_file = '/home/piaozx/文档/carla-code/carlafisheye/json2/'
    ann3_file = '/home/piaozx/文档/carla-code/carlafisheye/json3/'
    out1_dir = '/home/piaozx/文档/carla-code/carlafisheye/dis_pred1'
    out2_dir = '/home/piaozx/文档/carla-code/carlafisheye/dis_pred2'
    out3_dir = '/home/piaozx/文档/carla-code/carlafisheye/dis_pred3'
    bev1_file = '/home/piaozx/文档/carla-code/carlafisheye/dis_bev1/'
    bev2_file = '/home/piaozx/文档/carla-code/carlafisheye/dis_bev2/'
    bev3_file = '/home/piaozx/文档/carla-code/carlafisheye/dis_bev3/'
    # 遍历图像文件夹中的图像
    camera = ["fishf", "fishb", "fishr", "fishl"]
    
    disflag = 0
    for i in range(len(camera)):
        cam = camera[i]
        frame = "frame" + str(n)
        # 鱼眼图像检测
        img_path = image1_file + cam + '/' + frame + ".png"
        ann_path = ann1_file + cam + '/' + frame + '.json'
        result1, data1 = inference_mono_3d_detector(model, img_path, ann_path)
        return_out_dir, return_file_name, boundingboxes1 = show_result_meshlab(data1, result1, out1_dir, bev1_file, cam, ego_loc1, disflag, show=False, score_thr=0.25, task='mono-det')
        disflag += 1
    box1 = copy.deepcopy(boundingboxes1)

    disflag = 0
    for i in range(len(camera)):
        cam = camera[i]
        frame = "frame" + str(n)
        # 鱼眼图像检测
        img_path = image2_file + cam + '/' + frame + ".png"
        ann_path = ann2_file + cam + '/' + frame + '.json'
        result2, data2 = inference_mono_3d_detector(model, img_path, ann_path)
        return_out_dir, return_file_name, boundingboxes2 = show_result_meshlab(data2, result2, out2_dir, bev2_file, cam, ego_loc2, disflag, show=False, score_thr=0.25, task='mono-det')
        disflag += 1
    box2 = copy.deepcopy(boundingboxes2)

    disflag = 0
    for i in range(len(camera)):
        cam = camera[i]
        frame = "frame" + str(n)
        # 鱼眼图像检测
        img_path = image3_file + cam + '/' + frame + ".png"
        ann_path = ann3_file + cam + '/' + frame + '.json'
        result3, data3 = inference_mono_3d_detector(model, img_path, ann_path)
        return_out_dir, return_file_name, boundingboxes3 = show_result_meshlab(data3, result3, out3_dir, bev3_file, cam, ego_loc3, disflag, show=False, score_thr=0.25, task='mono-det')
        disflag += 1
    box3 = copy.deepcopy(boundingboxes3)
    
    return box1, box2, box3



def detection3d(n, ego_loc, image_file, ann_file, out_dir, bev_file):
    boundingboxes = []
    model = init_model(config_file, checkpoints_file, device='cuda:0')

    # 额外设置变量方便调试用
    # image_file = '/home/piaozx/文档/carla-code/carlafisheye/output/'
    # ann_file = '/home/piaozx/文档/carla-code/carlafisheye/json/'
    # out_dir = '/home/piaozx/文档/carla-code/carlafisheye/dis_pred'
    # bev_file = '/home/piaozx/文档/carla-code/carlafisheye/dis_bev/'
    
    # 遍历图像文件夹中的图像
    camera = ["fishf", "fishb", "fishr", "fishl"]    
    disflag = 0
    for i in range(len(camera)):
        cam = camera[i]
        frame = "frame" + str(n)
        # 鱼眼图像检测
        img_path = image_file + cam + '/' + frame + ".png"
        ann_path = ann_file + cam + '/' + frame + '.json'
        result1, data1 = inference_mono_3d_detector(model, img_path, ann_path)
        return_out_dir, return_file_name, boundingboxes = show_result_meshlab(data1, result1, out_dir, bev_file, cam, ego_loc, disflag, show=False, score_thr=0.25, task='mono-det')
        disflag += 1
    return boundingboxes

# def monodet3d_undistort(n, ego_loc):
#     boundingboxes = []
#     model = init_model(config_file, checkpoints_file, device='cuda:0')
#     und_image_file = '/home/piaozx/文档/carla-code/carlafisheye/und/'
#     ann_file = '/home/piaozx/文档/carla-code/carlafisheye/json/'
#     und_out_dir = '/home/piaozx/文档/carla-code/carlafisheye/und_pred'
#     und_bev_file = '/home/piaozx/文档/carla-code/carlafisheye/und_bev/'
#     # 遍历图像文件夹中的图像
#     camera = ["fishf", "fishb", "fishr", "fishl"]

#     undflag = 0
#     for i in range(len(camera)):
#         cam = camera[i]
#         frame = "frame" + str(n)
#         # 矫正图像检测
#         und_img_path = und_image_file + cam + '/' + frame + ".png"
#         ann_path = ann_file + cam + '/' + frame + '.json'
#         result2, data2 = inference_mono_3d_detector(model, und_img_path, ann_path)
#         return_out_dir, return_file_name, boundingboxes = show_result_meshlab(data2, result2, und_out_dir, und_bev_file, cam, ego_loc, undflag, score_thr=0.25, show=False, task='mono-det')
#         undflag += 1
#     return boundingboxes

# pgd_2xnux模型 score_thr=0.4
# config_file = '/home/piaozx/mmdetection3d-conda/configs/pgd/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d.py'
# checkpoints_file = '/home/piaozx/mmdetection3d-conda/checkpoints/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_20211112_125314-cb677266.pth'

    