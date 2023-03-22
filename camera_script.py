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
        if i == 0:
            mkdir_folder(args.save_path, 'bottom')
            filename = args.save_path +'bottom/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 1:
            mkdir_folder(args.save_path, 'top')
            filename = args.save_path +'top/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 2:
            mkdir_folder(args.save_path, 'left')
            filename = args.save_path +'left/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 3:
            mkdir_folder(args.save_path, 'right')
            filename = args.save_path +'right/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))
        elif i == 4:
            mkdir_folder(args.save_path, 'front')
            filename = args.save_path +'front/'+str(w_frame).zfill(6)+'.png'
            cv2.imwrite(filename, np.array(rgbs[i][...,::-1]))

def fisheye_stitch(n):
    """
    Loads input images from the path specified and creates a list of
    four-stream graphs generated with the imported images.
    """
    
    print('\nGenerating four-stream graphs...\n')

    
    for i in tqdm(range(n)):  # Tqdm：快速,可扩展的Python进度条,可以在 Python 长循环中添加一个进度提示信息
    
        output_image = np.zeros((2688,2688,3))

        front = cv2.imread('./output/fishf/frame' + str(i) + '.png')
        right = cv2.imread('./output/fishr/frame' + str(i) + '.png')
        back = cv2.imread('./output/fishb/frame' + str(i) + '.png')
        left = cv2.imread('./output/fishl/frame' + str(i) + '.png')

        # 将4张鱼眼图片拼接成四流图的样子，并保存至output_image变量中
        if not front is None:
        
            h = front.shape[0] # 1344
            w = front.shape[1] # 1344

            output_image[0:h, 0:w] = front
            output_image[0:h, w:w+w] = back
            output_image[h:h+h, 0:w] = left
            output_image[h:h+h, w:w+w] = right

            cv2.imwrite('./outcome/frame' + str(i) + '.png', output_image)

def mkdir_folder(path, name):
    if not os.path.isdir(os.path.join(path, name)):   # os.path.join() 用于路径拼接文件路径，可以传入多个路径
        os.makedirs(os.path.join(path, name))
    return True

# modify from manual control 将图像整改为numpy数组形式
def _parse_image_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3] # 取三维数组array中0到2维所有数据(应该是只有rgb值的所有像素)
    array = array[:, :, ::-1]
    return array

# modify from world on rail code
def visualize_data(rgb, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)):
    canvas = np.array(rgb[...,::-1])
    return canvas

def create_bevimage(image, x0, z0, yaw, l, w, frame, i):
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
    mkdir_folder("/home/piaozx/文档/carla-code/cube2test/", "bev")
    cv2.imwrite('/home/piaozx/文档/carla-code/cube2test/bev/frame'+str(frame).zfill(6)+'.png', image)
    # return image
    

def create_bevimages(image, x0, z0, yaw, frame):
    # 原点坐标
    ego_x = 512
    ego_z = 256
    l = 5
    w = 2
    
    font = cv2.FONT_HERSHEY_COMPLEX
    for i in range(0, len(x0)):
        x0[i] *= 10
        z0[i] *= 10
        l *= 10
        w *= 10
        yaw[i] = yaw[i]/180*math.pi
        x00 = x0[i] + w*math.cos(yaw[i])/2
        z00 = z0[i] + w*math.sin(yaw[i])/2
        x4 = x00 - l*math.sin(yaw[i])/2
        z4 = z00 + l*math.cos(yaw[i])/2
        x3 = x4 + l*math.sin(yaw[i])
        z3 = z4 - l*math.cos(yaw[i])
        x2 = x3 - w*math.cos(yaw[i])
        z2 = z3 - w*math.sin(yaw[i])
        x1 = x2 - l*math.sin(yaw[i])
        z1 = z2 + l*math.cos(yaw[i])

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
        image = cv2.line(image, (x1,z1), (x2,z2), (255,0,0), 1)
        image = cv2.line(image, (x1,z1), (x4,z4), (255,0,0), 1)
        image = cv2.line(image, (x3,z3), (x2,z2), (255,0,0), 1)
        image = cv2.line(image, (x3,z3), (x4,z4), (255,0,0), 1)
        cv2.putText(image, str(i+1), (int(x1), int(z1)), font, 0.5, (255,0,0), 1, cv2.LINE_AA)

    # return image

    filename = '/home/piaozx/文档/carla-code/cube2test/bev/frame'+str(frame).zfill(6)+'.png'
    cv2.imwrite(filename, image)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

def create_bev_box(image, x1, x2, x3, x4, z1, z2, z3, z4):
    
    image = cv2.line(image, (x1,z1), (x2,z2), (255,0,0), 1)
    image = cv2.line(image, (x1,z1), (x4,z4), (255,0,0), 1)
    image = cv2.line(image, (x3,z3), (x2,z2), (255,0,0), 1)
    image = cv2.line(image, (x3,z3), (x4,z4), (255,0,0), 1)

    return image