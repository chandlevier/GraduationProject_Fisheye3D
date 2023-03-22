'''
终端运行程序时需要输入图像文件和标注文件相对于mmdetection3d-conda文件夹的路径
例如:python3 my_demo/monodet3d.py my_input/checkerboard/xx.jpg my_input/checkerboard/xx.json
'''
from argparse import ArgumentParser
from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)
import glob
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('image_file', help='image files')
    parser.add_argument('ann_file', help='ann files')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    # pgd_2xnux模型 score_thr=0.4
    # config_file = 'configs/pgd/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d.py'
    # checkpoints_file = 'checkpoints/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_20211112_125314-cb677266.pth'
    # fcos3d模型 score_thr=0.2
    config_file = 'configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py'
    checkpoints_file = 'checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'
    
    model = init_model(config_file, checkpoints_file, device='cuda:0')
    # test a single image
    # 遍历图像文件夹中的图像
    # image_file=~/文档/carla-code/cube2test/output/
    images = glob.glob(args.image_file + '*.jpg')
    for fname in images:
        img_path = args.image_file + os.path.basename(fname)
        ann_path = args.ann_file + os.path.splitext(os.path.basename(fname))[0] + '.json'
        result, data = inference_mono_3d_detector(model, img_path, ann_path)
    # show the results
    # 输出结果可视化
    out_dir = '~/文档/carla-code/cube2test/pred'
    show_result_meshlab(data, result, out_dir, show=False, score_thr=0.25, task='mono-det')

if __name__ == '__main__':
    main()
