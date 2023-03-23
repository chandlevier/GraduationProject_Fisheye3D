import carla
import os
from queue import Queue, Empty
import cv2
import numpy as np
from tqdm import tqdm
import math

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

def rgbcams_store(sensor_list, args, w_frame, rgbs):
    for i in range (0, len(sensor_list)):
        # 左侧相机, bottom-top-left-right-front
        if i == 0:
            mkdir_folder(args.save_path, 'fishl', 'bottom')
            filename = args.save_path +'fishl/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 1:
            mkdir_folder(args.save_path, 'fishl', 'top')
            filename = args.save_path +'fishl/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 2:
            mkdir_folder(args.save_path, 'fishl', 'left')
            filename = args.save_path +'fishl/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 3:
            mkdir_folder(args.save_path, 'fishl', 'right')
            filename = args.save_path +'fishl/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 4:
            mkdir_folder(args.save_path, 'fishl', 'front')
            filename = args.save_path +'fishl/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 后侧相机, bottom-top-left-right-front
        elif i == 5:
            mkdir_folder(args.save_path, 'fishb', 'bottom')
            filename = args.save_path +'fishb/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 6:
            mkdir_folder(args.save_path, 'fishb', 'top')
            filename = args.save_path +'fishb/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 7:
            mkdir_folder(args.save_path, 'fishb', 'left')
            filename = args.save_path +'fishb/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 8:
            mkdir_folder(args.save_path, 'fishb', 'right')
            filename = args.save_path +'fishb/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 9:
            mkdir_folder(args.save_path, 'fishb', 'front')
            filename = args.save_path +'fishb/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        # 右侧相机, bottom-top-left-right-front
        elif i == 10:
            mkdir_folder(args.save_path, 'fishr', 'bottom')
            filename = args.save_path +'fishr/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 11:
            mkdir_folder(args.save_path, 'fishr', 'top')
            filename = args.save_path +'fishr/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 12:
            mkdir_folder(args.save_path, 'fishr', 'left')
            filename = args.save_path +'fishr/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 13:
            mkdir_folder(args.save_path, 'fishr', 'right')
            filename = args.save_path +'fishr/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 14:
            mkdir_folder(args.save_path, 'fishr', 'front')
            filename = args.save_path +'fishr/front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1])) 
        # 前侧相机, bottom-top-left-right-front
        elif i == 15:
            mkdir_folder(args.save_path, 'fishf', 'bottom')
            filename = args.save_path +'fishf/bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 16:
            mkdir_folder(args.save_path, 'fishf', 'top')
            filename = args.save_path +'fishf/top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 17:
            mkdir_folder(args.save_path, 'fishf', 'left')
            filename = args.save_path +'fishf/left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 18:
            mkdir_folder(args.save_path, 'fishf', 'right')
            filename = args.save_path +'fishf/right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 19:
            mkdir_folder(args.save_path, 'fishf', 'front')
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
def visualize_data(rgb, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)):
    canvas = np.array(rgb[...,::-1])
    return canvas

# modify from manual control 将图像整改为numpy数组形式
def _parse_image_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3] # 取三维数组array中0到2维所有数据(应该是只有rgb值的所有像素)
    array = array[:, :, ::-1]
    return array


def create_bev_groundtruth(image, x0, z0, yaw, l, w, frame, i):
    # print("调用成功！")
    # 原点坐标
    ego_x = 256
    ego_z = 512
    # l = 2
    # w = 5
    
    x0 *= 10
    z0 *= 10
    l *= 10
    w *= 10
    yaw = yaw/180*math.pi
    x00 = x0 + w*math.cos(yaw)/2
    z00 = z0 + w*math.sin(yaw)/2
    x4 = x00 - l*math.sin(yaw)/2
    z4 = z00 + l*math.cos(yaw)/2
    x3 = x4 + l*math.sin(yaw)
    z3 = z4 - l*math.cos(yaw)
    x2 = x3 - w*math.cos(yaw)
    z2 = z3 - w*math.sin(yaw)
    x1 = x2 - l*math.sin(yaw)
    z1 = z2 + l*math.cos(yaw)

    x1, x2, x3, x4 = int(x1), int(x2), int(x3), int(x4)
    z1, z2, z3, z4 = int(z1), int(z2), int(z3), int(z4)

    x1 += ego_x
    x2 += ego_x
    x3 += ego_x
    x4 += ego_x
    z1 = ego_z - z1
    z2 = ego_z - z2
    z3 = ego_z - z3
    z4 = ego_z - z4

    # img = cv2.rectangle(image, (x1,z1), (x3,z3), (255,0,0), 3)
    # image = create_bev_box(image, x1, x2, x3, x4, z1, z2, z3, z4)
    image = cv2.line(image, (x1,z1), (x2,z2), (0,0,255), 1)
    image = cv2.line(image, (x1,z1), (x4,z4), (0,0,255), 1)
    image = cv2.line(image, (x3,z3), (x2,z2), (0,0,255), 1)
    image = cv2.line(image, (x3,z3), (x4,z4), (0,0,255), 1)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image, str(i+1), (int(x1), int(z1)), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
    # 存储鸟瞰真值
    mkdir_folder("/home/piaozx/文档/carla-code/cube2fisheye/", "bev_groundtruth", None)
    cv2.imwrite('/home/piaozx/文档/carla-code/cube2fisheye/bev_groundtruth/frame'+str(frame).zfill(6)+'.png', image)
    # return image

def create_bev_box(image, x1, x2, x3, x4, z1, z2, z3, z4):
    
    image = cv2.line(image, (x1,z1), (x2,z2), (0,0,255), 1)
    image = cv2.line(image, (x1,z1), (x4,z4), (0,0,255), 1)
    image = cv2.line(image, (x3,z3), (x2,z2), (0,0,255), 1)
    image = cv2.line(image, (x3,z3), (x4,z4), (0,0,255), 1)

    return image
