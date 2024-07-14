#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# 单纯在车上安装一个rgb相机，然后实时将相机图像转换为鱼眼图像，并用pygame显示出来
# 问题：六个相机之间不同步，会掉帧
import glob
import os
import sys
import numpy as np
import pygame
import carla
import random
import time
import main_script

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


IM_WIDTH = 1024
IM_HEIGHT = 1024

def process_image_front(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # 把缓冲区(buffer)image.raw_data转变为一维阵列array
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))            # 把一维阵列array转换为行为image.height、列为image.width的矩阵
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # 创建一个与数组上的数据和格式最相似的新Surface。数组可以是2D或3D，具有任意大小的整数值。
    my_display.blit(surface, (0, 0))
    image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/front/%06d.png' % image.frame)
    return surface
def process_image_right(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # 把缓冲区(buffer)image.raw_data转变为一维阵列array
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))            # 把一维阵列array转换为行为image.height、列为image.width的矩阵
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # 创建一个与数组上的数据和格式最相似的新Surface。数组可以是2D或3D，具有任意大小的整数值。
    my_display.blit(surface, (0, 0))
    image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/right/%06d.png' % image.frame)
    return surface
def process_image_left(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # 把缓冲区(buffer)image.raw_data转变为一维阵列array
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))            # 把一维阵列array转换为行为image.height、列为image.width的矩阵
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # 创建一个与数组上的数据和格式最相似的新Surface。数组可以是2D或3D，具有任意大小的整数值。
    my_display.blit(surface, (0, 0))
    image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/left/%06d.png' % image.frame)
def process_image_back(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # 把缓冲区(buffer)image.raw_data转变为一维阵列array
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))            # 把一维阵列array转换为行为image.height、列为image.width的矩阵
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # 创建一个与数组上的数据和格式最相似的新Surface。数组可以是2D或3D，具有任意大小的整数值。
    my_display.blit(surface, (0, 0))
    image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/back/%06d.png' % image.frame)
def process_image_top(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # 把缓冲区(buffer)image.raw_data转变为一维阵列array
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))            # 把一维阵列array转换为行为image.height、列为image.width的矩阵
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # 创建一个与数组上的数据和格式最相似的新Surface。数组可以是2D或3D，具有任意大小的整数值。
    my_display.blit(surface, (0, 0))
    image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/top/%06d.png' % image.frame)
def process_image_bottom(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # 把缓冲区(buffer)image.raw_data转变为一维阵列array
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))            # 把一维阵列array转换为行为image.height、列为image.width的矩阵
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # 创建一个与数组上的数据和格式最相似的新Surface。数组可以是2D或3D，具有任意大小的整数值。
    my_display.blit(surface, (0, 0))
    image.save_to_disk('/home/piaozx/文档/carla-code/cube2fisheye/input/bottom/%06d.png' % image.frame)

actor_list = []


try:
    # #######################################################
    # === 1, 创建pygame窗口 ===
    pygame.init()
    size = IM_WIDTH, IM_HEIGHT
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("video")
    my_display = pygame.display.set_mode(size, pygame.HWSURFACE | pygame.DOUBLEBUF)
    # pygame_clock = pygame.time.Clock()
    
    
    # #######################################################
    # === 2, 在车辆上创建摄像头获得图像 ===
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    
    blueprint_library = world.get_blueprint_library()
    # 搞个小车
    vehicle_bp = blueprint_library.find('vehicle.mercedes-benz.coupe')
    vehicle_bp.set_attribute('color', '0, 0, 0')
    vehicle_transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
    vehicle.set_autopilot(True)
    actor_list.append(vehicle)
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    
    # create directory for outputs
    output_path = '/home/piaozx/文档/carla-code/cube2fisheye/input'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    #change the dimension of the image
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "90")
    
    #camera_bp.set_attribute('sensor_tick', '1.0')
    
    #设定传感器的相对位置(x方向偏移2.5，z方向偏移0.7，y方向偏移)
    #调整传感器的角度可在carla.Transform里添加carla.Rotation(roll,pitch,yew),分别代表x,y,z轴
    #不设置角度当前传感器与车头前向保持一致
    camera01_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=0))   # front
    camera02_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=90))  # right
    camera03_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=180)) # back
    camera04_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=270)) # left
    camera05_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=90))   # top
    camera06_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=-90))  # bottom
    #将传感器附在小车上
    camera01 = world.spawn_actor(camera_bp, camera01_transform, attach_to=vehicle)
    camera02 = world.spawn_actor(camera_bp, camera02_transform, attach_to=vehicle)
    camera03 = world.spawn_actor(camera_bp, camera03_transform, attach_to=vehicle)
    camera04 = world.spawn_actor(camera_bp, camera04_transform, attach_to=vehicle)
    camera05 = world.spawn_actor(camera_bp, camera05_transform, attach_to=vehicle)
    camera06 = world.spawn_actor(camera_bp, camera06_transform, attach_to=vehicle)
    #传感器开始监听
    camera01.listen(lambda image: process_image_front(image))
    camera02.listen(lambda image: process_image_right(image))
    camera03.listen(lambda image: process_image_back(image))
    camera04.listen(lambda image: process_image_left(image))
    camera05.listen(lambda image: process_image_top(image))
    camera06.listen(lambda image: process_image_bottom(image))

    actor_list.append(camera01)
    actor_list.append(camera02)
    actor_list.append(camera03)
    actor_list.append(camera04)
    actor_list.append(camera05)
    actor_list.append(camera06)


    
    # #######################################################
    # === 3, 设定观察者位置，保证小车一直在视图内 ===
    # spectator
    while True:
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),carla.Rotation(pitch=-90)))
        # main_script()
        # pygame.display.update()  # 对pygame显示窗口进行更新，默认窗口全部重绘
        time.sleep(0.005)
        for event in pygame.event.get():  # 遍历所有用户操作事件
            if event.type == pygame.QUIT:  # 获得事件类型，判断是否为关闭窗口
                sys.exit()   # 用于结束程序的运行并退出

    
    
finally:
    pygame.quit()  # 退出pygame
    print("All cleaned up!:)")
    for actor in actor_list:
    	actor.destroy()
    
    
