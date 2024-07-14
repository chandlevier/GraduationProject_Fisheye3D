#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# 在车上安装4个rgb相机，然后实时将相机图像转换为鱼眼图像,拼接起来并存储

import glob
import os
import sys
import time
import carla
import random
import numpy as np
import cv2
from queue import Queue, Empty
import copy
import random
import pygame
# random.seed(0)

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# argument设置
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--host', metavar='H',    default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
parser.add_argument('--port', '-p',           default=2000, type=int, help='TCP port to listen to (default: 2000)')
parser.add_argument('--tm_port',              default=8000, type=int, help='Traffic Manager Port (default: 8000)')
parser.add_argument('--ego-spawn', type=list, default=None, help='[x,y] in world coordinate')
parser.add_argument('--top-view',             default=True, help='Setting spectator to top view on ego car')
parser.add_argument('--map',                  default='Town04', help='Town Map')
parser.add_argument('--sync',                 default=True, help='Synchronous mode execution')
parser.add_argument('--sensor-h',             default=2.4, help='Sensor Height')
parser.add_argument('--save-path',            default='/home/piaozx/文档/carla-code/cube2fisheye/input/', help='Synchronous mode execution')
args = parser.parse_args()

# 图片大小
IM_WIDTH = 1024
IM_HEIGHT = 1024

pygame.init()
size = IM_WIDTH, IM_HEIGHT
screen = pygame.display.set_mode(size)
pygame.display.set_caption("video")
my_display = pygame.display.set_mode(size, pygame.HWSURFACE | pygame.DOUBLEBUF)

def process_image(image, i):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # 把缓冲区(buffer)image.raw_data转变为一维阵列array
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))            # 把一维阵列array转换为行为image.height、列为image.width的矩阵
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # 创建一个与数组上的数据和格式最相似的新Surface。数组可以是2D或3D，具有任意大小的整数值。
    my_display.blit(surface, (0, 0))
    if i == 0:
        image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/front/%06d.png' % image.frame)
    elif i == 1:
        image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/right/%06d.png' % image.frame)
    elif i == 2:
        image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/back/%06d.png' % image.frame)
    elif i == 3:
        image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/left/%06d.png' % image.frame)
    elif i == 4:
        image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/top/%06d.png' % image.frame)
    elif i == 5:
        image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/bottom/%06d.png' % image.frame)

actor_list, sensor_list = [], []
sensor_type = ['rgb']
def main(args):
    # 创造client
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)   
    world = client.get_world()
    # world = client.load_world('Town01')
    blueprint_library = world.get_blueprint_library()
    
    try:
        
        # 获取原有模式设置,退出时要还原模式设置
        original_settings = world.get_settings()
        settings = world.get_settings()

        # 设置自动模式
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()

        # 创建车辆
        ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # 手动规定
        # transform_vehicle = carla.Transform(carla.Location(0, 10, 0), carla.Rotation(0, 0, 0))
        # 随机选择
        transform_vehicle = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform_vehicle)
        actor_list.append(ego_vehicle)

        #--------------------------- 传感器部分 ---------------------------#
        sensor_queue = Queue()
        cam_bp = blueprint_library.find('sensor.camera.rgb')

        # 设置相机参数
        cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        cam_bp.set_attribute("fov", "90")
        # cam_bp.set_attribute('sensor_tick', '0.1')
        # cam1_front
        cam1 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2.347, z=0.7),carla.Rotation(yaw=0)), attach_to=ego_vehicle)
        cam1.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_front"))
        sensor_list.append(cam1)
        # cam2_right
        cam2 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2.347, z=0.7),carla.Rotation(yaw=90)), attach_to=ego_vehicle)
        cam2.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_right"))
        sensor_list.append(cam2)
        # cam3_back
        cam3 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2.347, z=0.7),carla.Rotation(yaw=180)), attach_to=ego_vehicle)
        cam3.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_back"))
        sensor_list.append(cam3)
        # cam4_left
        cam4 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2.347, z=0.7),carla.Rotation(yaw=270)), attach_to=ego_vehicle)
        cam4.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_left"))
        sensor_list.append(cam4)
        # cam5_top
        cam5 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2.347, z=0.7),carla.Rotation(pitch=90)), attach_to=ego_vehicle)
        cam5.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_top"))
        sensor_list.append(cam5)
        # cam6_bottom
        cam6 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2.347, z=0.7),carla.Rotation(pitch=-90)), attach_to=ego_vehicle)
        cam6.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_bottom"))
        sensor_list.append(cam6)
        #-------------------------- 传感器设置完毕 --------------------------#

        # 设置traffic manager
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        # 是否忽略红绿灯
        tm.ignore_lights_percentage(ego_vehicle, 100)
        # 如果限速30km/h -> 30*(1+10%)=33km/h
        tm.global_percentage_speed_difference(-50.0)
        ego_vehicle.set_autopilot(True, tm.get_port())

        # from tutorial.py
        # 添加其他actor
        # transform_vehicle.location += carla.Location(x=20, y=-3.2)
        # transform_vehicle.rotation.yaw = -180.0
        # for _ in range(0, 10):
        #     transform_vehicle.location.x += 8.0

        #     bp = random.choice(blueprint_library.filter('vehicle'))

        #     # This time we are using try_spawn_actor. If the spot is already
        #     # occupied by another object, the function will return None.
        #     npc = world.try_spawn_actor(bp, transform_vehicle)
        #     if npc is not None:
        #         actor_list.append(npc)
        #         npc.set_autopilot(True)
        #         print('created %s' % npc.type_id)


        while True:
            # Tick the server
            world.tick()

            # 将CARLA界面摄像头跟随车动
            loc = ego_vehicle.get_transform().location
            spectator.set_transform(carla.Transform(carla.Location(x=loc.x,y=loc.y,z=35),carla.Rotation(yaw=0,pitch=-90,roll=0)))

            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)
            try:
                cam1.listen(lambda image: process_image(image,0))
                cam2.listen(lambda image: process_image(image,1))
                cam3.listen(lambda image: process_image(image,2))
                cam4.listen(lambda image: process_image(image,3))
                cam5.listen(lambda image: process_image(image,4))
                cam6.listen(lambda image: process_image(image,5))
                cv2.waitKey(100)
                # for i in range (0, len(sensor_list)):   # 遍历6个相机
                #     s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                #     print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
                #     if i == 0:
                #         filename = args.save_path + 'front/' + str(w_frame) + '.png'
                #         cv2.imwrite(filename, )
                    # sensor_type = s_name.split('_')[0]  # 只取"rgb_front"前面的"rgb"
                    # if sensor_type == 'rgb':
                    #     rgbs.append(_parse_image_cb(s_data))


                
                # 可视化
                # rgb=np.concatenate(rgbs, axis=1)[...,:3]
                # cv2.imshow('rgb', visualize_data(rgb))
                # cv2.waitKey(100)
                # if rgb is None or args.save_path is not None:
                #     # 检查是否有传感器的文件夹
                #     mkdir_folder(args.save_path)
                #     filename = args.save_path +'rgb/'+str(w_frame)+'.png'
                #     cv2.imwrite(filename, np.array(rgb[...,::-1]))

            except Empty:
                print("    Some of the sensor information is missed")

    finally:
        world.apply_settings(original_settings)
        tm.set_synchronous_mode(False)
        for sensor in sensor_list:
            sensor.destroy()
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!")

def mkdir_folder(path):
    for s_type in sensor_type:
        if not os.path.isdir(os.path.join(path, s_type)):
            os.makedirs(os.path.join(path, s_type))
    return True

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # 处理传感器数据,如:save_to_disk等,然后将传感器数据加入sensor_queue中
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))

# modify from world on rail code
def visualize_data(rgb, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)):
    canvas = np.array(rgb[...,::-1])
    return canvas

# modify from manual control
def _parse_image_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


if __name__ == "__main__":
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
