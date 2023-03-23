# Copyright (c) OpenMMLab. All rights reserved.
# 用于初始化模型、前向推理、读取图片、显示检测结果等
import re
from copy import deepcopy
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes,
                          show_multi_modality_result, show_result,
                          show_seg_result)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger
import math
import cv2


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.可以是'cpu', 'cuda:0', 'cuda:1'等

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        # 如果config是配置文件路径，则用mmcv.Config.fromfile()读取
        # 并返回类mmcv.Config()实例化后的object
        # 如果config已经是类mmcv.Config()实例化后的object，则不需要其他操作
        # 如果config不是配置文件路径，则报错
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    convert_SyncBN(config.model)
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = config.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    if device != 'cpu':
        torch.cuda.set_device(device)
    else:
        logger = get_root_logger()
        logger.warning('Don\'t suggest using CPU device. '
                       'Some functions are not supported for now.')
    model.to(device)
    model.eval()
    return model


def inference_detector(model, pcd):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if not isinstance(pcd, str):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadPointsFromDict'

    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

    if isinstance(pcd, str):
        # load from point clouds file
        data = dict(
            pts_filename=pcd,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            # for ScanNet demo we need axis_align_matrix
            ann_info=dict(axis_align_matrix=np.eye(4)),
            sweeps=[],
            # set timestamp = 0
            timestamp=[0],
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[])
    else:
        # load from http
        data = dict(
            points=pcd,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            # for ScanNet demo we need axis_align_matrix
            ann_info=dict(axis_align_matrix=np.eye(4)),
            sweeps=[],
            # set timestamp = 0
            timestamp=[0],
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def inference_multi_modality_detector(model, pcd, image, ann_file):
    """Inference point cloud with the multi-modality detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    data_infos = mmcv.load(ann_file)
    image_idx = int(re.findall(r'\d+', image)[-1])  # xxx/sunrgbd_000017.jpg
    for x in data_infos:
        if int(x['image']['image_idx']) != image_idx:
            continue
        info = x
        break
    data = dict(
        pts_filename=pcd,
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)

    # TODO: this code is dataset-specific. Move lidar2img and
    #       depth2img to .pkl annotations in the future.
    # LiDAR to image conversion
    if box_mode_3d == Box3DMode.LIDAR:
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        lidar2img = P2 @ rect @ Trv2c
        data['img_metas'][0].data['lidar2img'] = lidar2img
    # Depth to image conversion
    elif box_mode_3d == Box3DMode.DEPTH:
        rt_mat = info['calib']['Rt']
        # follow Coord3DMode.convert_point
        rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                           ]) @ rt_mat.transpose(1, 0)
        depth2img = info['calib']['K'] @ rt_mat
        data['img_metas'][0].data['depth2img'] = depth2img

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
        data['img'] = data['img'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def inference_mono_3d_detector(model, image, ann_file, bev_file, frame):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    data_infos = mmcv.load(ann_file)
    # find the info corresponding to this image
    for x in data_infos['images']:
        if osp.basename(x['file_name']) != osp.basename(image):
            continue
        img_info = x
        break
    data = dict(
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    # camera points to image conversion
    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(dict(cam_intrinsic=img_info['cam_intrinsic']))

    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    
    # 获取检测框的数量和角点位置,计算出检测框7个参数xyz、lwh、yaw
    # 具体步骤参考之前在image_vis.py中写的代码
    from mmdet3d.core.bbox import points_cam2img
    import copy
    img_filename = data['img_metas'][0][0]['filename']
    raw_img = mmcv.imread(img_filename)
    img = raw_img.copy()
    cam2img = copy.deepcopy(data['img_metas'][0][0]['cam2img'])

    if 'pts_bbox' in result[0].keys():
        result[0] = result[0]['pts_bbox']
    elif 'img_bbox' in result[0].keys():
        result[0] = result[0]['img_bbox']
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()


    bboxes3d = CameraInstance3DBoxes(
            pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    # 之前将所有角点整合到一个数组中，现在将不同bbox的角点分在一个数组的不同元素中
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()
    image = np.zeros((512, 512, 3), np.uint8)
    image.fill(255)
    font = cv2.FONT_HERSHEY_COMPLEX
    for i in range(num_bbox):
        x, y, z, l, w, h ,yaw = get_scale_and_center(corners_3d[i])
        create_bev(image, x, z, l, w, yaw, i, font)
    bev_filename = bev_file + frame + "_bev.png"
    cv2.imwrite(bev_filename, image)

    return result, data


def inference_segmentor(model, pcd):
    """Inference point cloud with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    data = dict(
        pts_filename=pcd,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def show_det_result_meshlab(data,
                            result,
                            out_dir,
                            score_thr=0.0,
                            show=False,
                            snapshot=False):
    """Show 3D detection result by meshlab."""
    points = data['points'][0][0].cpu().numpy()
    pts_filename = data['img_metas'][0][0]['pts_filename']
    file_name = osp.split(pts_filename)[-1].split('.')[0]

    if 'pts_bbox' in result[0].keys():
        pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
    else:
        pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['scores_3d'].numpy()

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    # for now we convert points into depth mode
    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if box_mode != Box3DMode.DEPTH:
        points = Coord3DMode.convert(points, box_mode, Coord3DMode.DEPTH)
        show_bboxes = Box3DMode.convert(pred_bboxes, box_mode, Box3DMode.DEPTH)
    else:
        show_bboxes = deepcopy(pred_bboxes)

    show_result(
        points,
        None,
        show_bboxes,
        out_dir,
        file_name,
        show=show,
        snapshot=snapshot)

    return file_name


def show_seg_result_meshlab(data,
                            result,
                            out_dir,
                            palette,
                            show=False,
                            snapshot=False):
    """Show 3D segmentation result by meshlab."""
    points = data['points'][0][0].cpu().numpy()
    pts_filename = data['img_metas'][0][0]['pts_filename']
    file_name = osp.split(pts_filename)[-1].split('.')[0]

    pred_seg = result[0]['semantic_mask'].numpy()

    if palette is None:
        # generate random color map
        max_idx = pred_seg.max()
        palette = np.random.randint(0, 256, size=(max_idx + 1, 3))
    palette = np.array(palette).astype(np.int)

    show_seg_result(
        points,
        None,
        pred_seg,
        out_dir,
        file_name,
        palette=palette,
        show=show,
        snapshot=snapshot)

    return file_name


def show_proj_det_result_meshlab(data,
                                 result,
                                 out_dir,
                                 score_thr=0.0,
                                 show=False,
                                 snapshot=False):
    """Show result of projecting 3D bbox to 2D image by meshlab."""
    assert 'img' in data.keys(), 'image data is not provided for visualization'

    img_filename = data['img_metas'][0][0]['filename']
    file_name = osp.split(img_filename)[-1].split('.')[0]

    # read from file because img in data_dict has undergone pipeline transform
    img = mmcv.imread(img_filename)

    if 'pts_bbox' in result[0].keys():
        result[0] = result[0]['pts_bbox']
    elif 'img_bbox' in result[0].keys():
        result[0] = result[0]['img_bbox']
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    pred_scores = result[0]['scores_3d'].numpy()

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if box_mode == Box3DMode.LIDAR:
        if 'lidar2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'LiDAR to image transformation matrix is not provided')

        show_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['lidar2img'],
            out_dir,
            file_name,
            box_mode='lidar',
            show=show)
    elif box_mode == Box3DMode.DEPTH:
        show_bboxes = DepthInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            None,
            out_dir,
            file_name,
            box_mode='depth',
            img_metas=data['img_metas'][0][0],
            show=show)
    elif box_mode == Box3DMode.CAM:
        if 'cam2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'camera intrinsic matrix is not provided')
        # 绘制出3Dbbox
        show_bboxes = CameraInstance3DBoxes(
            pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['cam2img'],
            out_dir,
            file_name,
            box_mode='camera',
            show=show)
    else:
        raise NotImplementedError(
            f'visualization of {box_mode} bbox is not supported')

    return file_name


def show_result_meshlab(data,
                        result,
                        out_dir,
                        score_thr=0.0,
                        show=False,
                        snapshot=False,
                        task='det',
                        palette=None):
    """Show result by meshlab.

    Args:
        data (dict): Contain data from pipeline.
        result (dict): Predicted result from model.
        out_dir (str): Directory to save visualized result.
        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.0
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results.
            Defaults to False.
        task (str, optional): Distinguish which task result to visualize.
            Currently we support 3D detection, multi-modality detection and
            3D segmentation. Defaults to 'det'.
        palette (list[list[int]]] | np.ndarray, optional): The palette
            of segmentation map. If None is given, random palette will be
            generated. Defaults to None.
    """
    assert task in ['det', 'multi_modality-det', 'seg', 'mono-det'], \
        f'unsupported visualization task {task}'
    assert out_dir is not None, 'Expect out_dir, got none.'

    if task in ['det', 'multi_modality-det']:
        file_name = show_det_result_meshlab(data, result, out_dir, score_thr,
                                            show, snapshot)

    if task in ['seg']:
        file_name = show_seg_result_meshlab(data, result, out_dir, palette,
                                            show, snapshot)

    if task in ['multi_modality-det', 'mono-det']:
        file_name = show_proj_det_result_meshlab(data, result, out_dir,
                                                 score_thr, show, snapshot)

    return out_dir, file_name





def get_scale_and_center(corner_3d):
    h = corner_3d[3][1] - corner_3d[0][1]
    l = pow(pow((corner_3d[1][0] - corner_3d[0][0]), 2) + pow((corner_3d[1][2] - corner_3d[0][2]), 2), 0.5)
    w = pow(pow((corner_3d[4][0] - corner_3d[0][0]), 2) + pow((corner_3d[4][2] - corner_3d[0][2]), 2), 0.5)
    # math.asin返回弧度,这里换算为角度
    yaw = (math.asin((corner_3d[4][2]-corner_3d[0][2])/w) / math.pi) * 180
    if yaw < 0:
        yaw = yaw + 180
    # 计算中心坐标的代码好像会修改到corner_3d数组,导致之后的长宽高计算不准确,所以需要先计算长宽高再计算坐标
    x = corner_3d[0][0]
    y = corner_3d[0][1]
    z = corner_3d[0][2]
    for i in range(7):
        x += corner_3d[i+1][0]
        y += corner_3d[i+1][1]
        z += corner_3d[i+1][2]
    x = x / 8
    y = y / 8
    z = z / 8

    return x, y, z, l, w, h, yaw

def create_bev(image, x0, z0, l, w, yaw, i, font):
    x0 *= 10
    z0 *= 10
    l *= 10
    w *= 10
    yaw = yaw/180*math.pi
    x00 = x0 + w*math.cos(yaw)/2
    z00 = z0 + w*math.sin(yaw)/2
    x3 = x00 - l*math.sin(yaw)/2
    z3= z00 + l*math.cos(yaw)/2
    x2 = x3 + l*math.sin(yaw)
    z2 = z3 - l*math.cos(yaw)
    x1 = x2 - w*math.cos(yaw)
    z1 = z2 - w*math.sin(yaw)
    x4 = x1 - l*math.sin(yaw)
    z4 = z1 + l*math.cos(yaw)

    x1, x2, x3, x4 = int(x1), int(x2), int(x3), int(x4)
    z1, z2, z3, z4 = int(z1), int(z2), int(z3), int(z4)

    x1 += 256
    x2 += 256
    x3 += 256
    x4 += 256
    z1 = 512 - z1
    z2 = 512 - z2
    z3 = 512 - z3
    z4 = 512 - z4
    image = cv2.line(image, (x1,z1), (x2,z2), (255,0,0), 1)
    image = cv2.line(image, (x1,z1), (x4,z4), (255,0,0), 1)
    image = cv2.line(image, (x3,z3), (x2,z2), (255,0,0), 1)
    image = cv2.line(image, (x3,z3), (x4,z4), (255,0,0), 1)
    cv2.putText(image, str(i+1), (int(x1), int(z1)), font, 0.5, (255,0,0), 1, cv2.LINE_AA)