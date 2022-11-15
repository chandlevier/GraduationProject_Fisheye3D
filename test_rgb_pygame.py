#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import numpy as np
import pygame

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time

IM_WIDTH = 640
IM_HEIGHT = 480

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    my_display.blit(surface, (0, 0))
    return surface




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
    
    vehicle_bp = blueprint_library.find('vehicle.mercedes-benz.coupe')
    vehicle_bp.set_attribute('color', '0, 0, 0')
    print(vehicle_bp)
    
    spawn_point = random.choice(world.get_map().get_spawn_points())
    
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)
    #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    
    actor_list.append(vehicle)
    
    
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    #change the dimension of the image
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "110")
    #camera_bp.set_attribute('sensor_tick', '1.0')
    
    camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    
    camera.listen(lambda image: process_image(image))
#    camera.listen(lambda image: image.save_to_disk(os.path.join(output_path, '%06d.png' % image.frame)))
    actor_list.append(camera)
    
    #spectator
    while True:
	    spectator = world.get_spectator()
	    transform = vehicle.get_transform()
	    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),carla.Rotation(pitch=-90)))
	    pygame.display.update()  # 对显示窗口进行更新，默认窗口全部重绘
	    time.sleep(0.01)

        
#    while True:  # 无限循环，确保窗口一直显示
#
#        for event in pygame.event.get():  # 遍历所有用户操作事件
#
#            if event.type == pygame.QUIT:  # 获得事件类型，判断是否为关闭窗口
#
#                sys.exit()   # 用于结束程序的运行并退出

        
    
    
    time.sleep(15)
    
    
finally:
    pygame.quit()  # 退出pygame
    for actor in actor_list:
    	actor.destroy()
    print("All cleaned up!:)")
    

