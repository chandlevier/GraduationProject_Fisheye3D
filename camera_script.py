import carla
import os
from queue import Queue, Empty
import cv2
import numpy as np
from tqdm import tqdm
import math
from mmdet3d.core.visualizer.image_vis import bev_corners

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    """
    处理传感器数据,如save_to_disk等,然后将传感器数据加入sensor_queue
    """
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))


def rgbcams_buildup(world, cam_bp, cam_location, ego_vehicle, sensor_list, sensor_queue, yaw0):
    """
    world: 主函数中创建的carla世界蓝图
    cam_bp: 主函数中设置的相机蓝图
    cam_location: 相机坐标，每辆车四个相机坐标不同
    ego_vehicle: 主函数中设置的车辆
    sensor_list: 传感器数组，每个鱼眼相机使用不同的传感器数组
    sensor_queue: 主函数中设置的传感器队列
    """
    # cam1_front
    cam1 = world.spawn_actor(cam_bp, carla.Transform(cam_location,carla.Rotation(yaw=yaw0+0)), attach_to=ego_vehicle)
    cam1.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_front"))
    sensor_list.append(cam1)
    # cam2_right
    cam2 = world.spawn_actor(cam_bp, carla.Transform(cam_location,carla.Rotation(yaw=yaw0+90)), attach_to=ego_vehicle)
    cam2.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_right"))
    sensor_list.append(cam2)
    # # cam3_back
    # cam3 = world.spawn_actor(cam_bp, carla.Transform(cam_location,carla.Rotation(yaw=yaw0+180)), attach_to=ego_vehicle)
    # cam3.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_back"))
    # sensor_list.append(cam3)
    # cam4_left
    cam4 = world.spawn_actor(cam_bp, carla.Transform(cam_location,carla.Rotation(yaw=yaw0+270)), attach_to=ego_vehicle)
    cam4.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_left"))
    sensor_list.append(cam4)
    # cam5_top
    cam5 = world.spawn_actor(cam_bp, carla.Transform(cam_location,carla.Rotation(yaw=yaw0, pitch=90)), attach_to=ego_vehicle)
    cam5.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_top"))
    sensor_list.append(cam5)
    # cam6_bottom
    cam6 = world.spawn_actor(cam_bp, carla.Transform(cam_location,carla.Rotation(yaw=yaw0, pitch=-90)), attach_to=ego_vehicle)
    cam6.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_bottom"))
    sensor_list.append(cam6)
    # sensor_list里的顺序是: bottom-top-left-(back-)right-front

def surfishcams_buildup(world, cam_bp, cam1_location, cam2_location, cam3_location, cam4_location, ego_vehicle, sensor_list, sensor_queue):
    # cam1:front cam2:right cam3:back cam4:left
    rgbcams_buildup(world, cam_bp, cam1_location, ego_vehicle, sensor_list, sensor_queue, yaw0=0)
    rgbcams_buildup(world, cam_bp, cam2_location, ego_vehicle, sensor_list, sensor_queue, yaw0=90)
    rgbcams_buildup(world, cam_bp, cam3_location, ego_vehicle, sensor_list, sensor_queue, yaw0=180)
    rgbcams_buildup(world, cam_bp, cam4_location, ego_vehicle, sensor_list, sensor_queue, yaw0=270)


def multirgbcams_store(sensor_list, args, w_frame, rgbs):
    mkdir_folder(args.save_path, 'vehicle1', None)
    mkdir_folder(args.save_path, 'vehicle2', None)
    mkdir_folder(args.save_path, 'vehicle3', None)
    input_savepath1 = args.save_path + 'vehicle1/'
    input_savepath2 = args.save_path + 'vehicle2/'
    input_savepath3 = args.save_path + 'vehicle3/'
    input_savepath = [input_savepath1, input_savepath2, input_savepath3]
    fishcams = ["fishf", "fishb", "fishr", "fishl"]
    cams = ["bottom", "top", "right", "left", "front"]
    # 创建所有透视图像输入文件夹
    for i in range(len(input_savepath)):
        for j in range(len(fishcams)):
            for k in range(len(cams)):
                mkdir_folder(input_savepath[i], fishcams[j], cams[k])

    
    for i in range (0, len(sensor_list)):
        # 车3左侧相机, bottom-top-left-right-front
        if i == 0:
            filename = input_savepath3 +'fishl/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 1:
            filename = input_savepath3 +'fishl/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 2:
            filename = input_savepath3 +'fishl/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 3:
            filename = input_savepath3 +'fishl/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 4:
            filename = input_savepath3 +'fishl/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 车3后侧相机, bottom-top-left-right-front
        elif i == 5:
            filename = input_savepath3 +'fishb/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 6:
            filename = input_savepath3 +'fishb/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 7:
            filename = input_savepath3 +'fishb/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 8:
            filename = input_savepath3 +'fishb/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 9:
            filename = input_savepath3 +'fishb/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 车3右侧相机, bottom-top-left-right-front
        elif i == 10:
            filename = input_savepath3 +'fishr/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 11:
            filename = input_savepath3 +'fishr/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 12:
            filename = input_savepath3 +'fishr/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 13:
            filename = input_savepath3 +'fishr/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 14:
            filename = input_savepath3 +'fishr/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1])) 
        # 车3前侧相机, bottom-top-left-right-front
        elif i == 15:
            filename = input_savepath3 +'fishf/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 16:
            filename = input_savepath3 +'fishf/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 17:
            filename = input_savepath3 +'fishf/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 18:
            filename = input_savepath3 +'fishf/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 19:
            filename = input_savepath3 +'fishf/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 车2左侧相机, bottom-top-left-right-front
        if i == 20:
            filename = input_savepath2 +'fishl/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 21:
            filename = input_savepath2 +'fishl/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 22:
            filename = input_savepath2 +'fishl/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 23:
            filename = input_savepath2 +'fishl/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 24:
            filename = input_savepath2 +'fishl/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 车2后侧相机, bottom-top-left-right-front
        elif i == 25:
            filename = input_savepath2 +'fishb/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 26:
            filename = input_savepath2 +'fishb/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 27:
            filename = input_savepath2 +'fishb/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 28:
            filename = input_savepath2 +'fishb/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 29:
            filename = input_savepath2 +'fishb/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 车2右侧相机, bottom-top-left-right-front
        elif i == 30:
            filename = input_savepath2 +'fishr/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 31:
            filename = input_savepath2 +'fishr/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 32:
            filename = input_savepath2 +'fishr/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 33:
            filename = input_savepath2 +'fishr/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 34:
            filename = input_savepath2 +'fishr/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1])) 
        # 车2前侧相机, bottom-top-left-right-front
        elif i == 35:
            filename = input_savepath2 +'fishf/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 36:
            filename = input_savepath2 +'fishf/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 37:
            filename = input_savepath2 +'fishf/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 38:
            filename = input_savepath2 +'fishf/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 39:
            filename = input_savepath2 +'fishf/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 车1左侧相机, bottom-top-left-right-front
        elif i == 40:
            filename = input_savepath1 +'fishl/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 41:
            filename = input_savepath1 +'fishl/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 42:
            filename = input_savepath1 +'fishl/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 43:
            filename = input_savepath1 +'fishl/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 44:
            filename = input_savepath1 +'fishl/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 车1后侧相机, bottom-top-left-right-front
        elif i == 45:
            filename = input_savepath1 +'fishb/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 46:
            filename = input_savepath1 +'fishb/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 47:
            filename = input_savepath1 +'fishb/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 48:
            filename = input_savepath1 +'fishb/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 49:
            filename = input_savepath1 +'fishb/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 车1右侧相机, bottom-top-left-right-front
        elif i == 50:
            filename = input_savepath1 +'fishr/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 51:
            filename = input_savepath1 +'fishr/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 52:
            filename = input_savepath1 +'fishr/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 53:
            filename = input_savepath1 +'fishr/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 54:
            filename = input_savepath1 +'fishr/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1])) 
        # 车1前侧相机, bottom-top-left-right-front
        elif i == 55:
            filename = input_savepath1 +'fishf/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 56:
            filename = input_savepath1 +'fishf/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 57:
            filename = input_savepath1 +'fishf/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 58:
            filename = input_savepath1 +'fishf/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 59:
            filename = input_savepath1 +'fishf/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1])) 

def rgbcams_store(sensor_list, args, w_frame, rgbs):
    fishcams = ["fishf", "fishb", "fishr", "fishl"]
    cams = ["bottom", "top", "right", "left", "front"]
    # 创建所有透视图像输入文件夹
    for i in range(len(fishcams)):
        for j in range(len(cams)):
            mkdir_folder(args.save_path, fishcams[i], cams[j])

    
    for i in range (0, len(sensor_list)):
        # 车2左侧相机, bottom-top-left-right-front
        if i == 0:
            filename = args.save_path +'fishl/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 1:
            filename = args.save_path +'fishl/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 2:
            filename = args.save_path +'fishl/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 3:
            filename = args.save_path +'fishl/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 4:
            filename = args.save_path +'fishl/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 车2后侧相机, bottom-top-left-right-front
        elif i == 5:
            filename = args.save_path +'fishb/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 6:
            filename = args.save_path +'fishb/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 7:
            filename = args.save_path +'fishb/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 8:
            filename = args.save_path +'fishb/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 9:
            filename = args.save_path +'fishb/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 车2右侧相机, bottom-top-left-right-front
        elif i == 10:
            filename = args.save_path +'fishr/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 11:
            filename = args.save_path +'fishr/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 12:
            filename = args.save_path +'fishr/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 13:
            filename = args.save_path +'fishr/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 14:
            filename = args.save_path +'fishr/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1])) 
        # 车2前侧相机, bottom-top-left-right-front
        elif i == 15:
            filename = args.save_path +'fishf/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 16:
            filename = args.save_path +'fishf/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 17:
            filename = args.save_path +'fishf/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 18:
            filename = args.save_path +'fishf/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 19:
            filename = args.save_path +'fishf/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))

def fisheye_stitch(n):
    """
    Loads input images from the path specified and creates a list of
    four-stream graphs generated with the imported images.
    """
    
    print('\nGenerating four-stream graphs...\n')

    
    for i in tqdm(range(n)):  # Tqdm：快速,可扩展的Python进度条,可以在 Python 长循环中添加一个进度提示信息
    
        output_image = np.zeros((2688,2688,3))

        front = cv2.imread('./output/frame' + str(i) + '.png')
        right = cv2.imread('./output/frame' + str(i) + '.png')
        back = cv2.imread('./output/frame' + str(i) + '.png')
        left = cv2.imread('./output/frame' + str(i) + '.png')

        # 将4张鱼眼图片拼接成四流图的样子，并保存至output_image变量中
        if not front is None:
        
            h = front.shape[0] # 1344
            w = front.shape[1] # 1344

            output_image[0:h, 0:w] = front
            output_image[0:h, w:w+w] = back
            output_image[h:h+h, 0:w] = left
            output_image[h:h+h, w:w+w] = right

            cv2.imwrite('./outcome/frame' + str(i) + '.png', output_image)


def mkdir_folder(path, name1, name2):
    if name2 is not None:
        if not os.path.isdir(os.path.join(path, name1, name2)):
            os.makedirs(os.path.join(path, name1, name2))
    else:
        if not os.path.isdir(os.path.join(path, name1)):
            os.makedirs(os.path.join(path, name1))
    return True

# modify from world on rail code
def visualize_data(rgb, text_args=(cv2.FONT_HERSHEY_COMPLEX, 0.3, (255,255,255), 1)):
    canvas = np.array(rgb[...,::-1])
    return canvas

# modify from manual control 将图像整改为numpy数组形式
def _parse_image_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3] # 取三维数组array中0到2维所有数据(应该是只有rgb值的所有像素)
    array = array[:, :, ::-1]
    return array

# 真值不需要相机参数
groundtruths = []
class groundtruth():
    def __init__(self):
        self.num = ""
        self.frame = ""
        self.x = ""
        self.z = ""
        self.l = ""
        self.w = ""
        self.yaw = ""


# 根据carla中actor的位置生成鸟瞰真值
def create_bev_groundtruth(image, x0, z0, yaw, l, w, frame, i, box_class, color=(0,0,255)):
    # 原点坐标
    # ego_x = 512
    # ego_z = 512
    x0 *= 10
    z0 *= 10
    l *= 10
    w *= 10
    yaw = yaw/180*math.pi
    u1, u2, u3, u4, v1, v2, v3, v4 = bev_corners(x0, z0, l, w, yaw, 512, 768)
    if i > 0:
        box_class.bev_coord = u1, u2, u3, u4, v1, v2, v3, v4
    # image = cv2.line(image, (u1,v1), (u2,v2), (0,0,255), 1)
    # image = cv2.line(image, (u1,v1), (u4,v4), (0,0,255), 1)
    # image = cv2.line(image, (u3,v3), (u2,v2), (0,0,255), 1)
    # image = cv2.line(image, (u3,v3), (u4,v4), (0,0,255), 1)
    image = create_bev_box(image, u1, u2, u3, u4, v1, v2, v3, v4, color,)
    font = cv2.FONT_HERSHEY_COMPLEX
    font_u, font_v = find_northwest_point(u1, u2, u3, u4, v1, v2, v3, v4)
    cv2.putText(image, str(i), (int(font_u)-10, int(font_v)), font, 0.5, color, 1, cv2.LINE_AA)
    # 存储鸟瞰真值
    mkdir_folder("/home/piaozx/文档/carla-code/carlafisheye/", "bev_groundtruth", None)
    cv2.imwrite('/home/piaozx/文档/carla-code/carlafisheye/bev_groundtruth/frame'+str(frame).zfill(6)+'.png', image)
    # return image

def create_bev_box(image, x1, x2, x3, x4, z1, z2, z3, z4, color=(0,0,255)):
    
    image = cv2.line(image, (x1,z1), (x2,z2), color, 1)
    image = cv2.line(image, (x1,z1), (x4,z4), color, 1)
    image = cv2.line(image, (x3,z3), (x2,z2), color, 1)
    image = cv2.line(image, (x3,z3), (x4,z4), color, 1)
    if abs(x2-x1) < 5:  # 如果x1、x2都在一侧
        row_start = min(z1, z2)
        row_end = max(z1, z2)
        for row in range(row_start, row_end+1):
            image = cv2.line(image, (x1,row), (x3,row), color, 1)
    elif abs(x1-x4) < 5:
        row_start = min(z1, z4)
        row_end = max(z1, z4)
        for row in range(row_start, row_end+1):
            image = cv2.line(image, (x1,row), (x2,row), color, 1)

    return image

def find_northwest_point(u1, u2, u3, u4, v1, v2, v3, v4):
    u = [u1, u2, u3, u4]
    v = [v1, v2, v3, v4]
    area_min = u[0] * v[0]
    min_num = 0
    for i in range(1, 4):
        area = u[i] * v[i]
        if area < area_min:
            area_min = area
            min_num = i
    return u[min_num], v[min_num]


