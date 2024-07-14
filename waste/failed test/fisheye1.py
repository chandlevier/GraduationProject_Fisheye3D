#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# 在车上安装1个rgb相机，然后实时将相机图像转换为鱼眼图像

import glob
import os
import sys
import numpy as np
import pygame
import carla
import random
import time
import cv2
from queue import Queue, Empty
random.seed(10)
import main_script


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


IM_WIDTH = 640
IM_HEIGHT = 640

# def process_image(image):
#     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # 把缓冲区(buffer)image.raw_data转变为一维阵列array
#     array = np.reshape(array, (IM_HEIGHT, IM_WIDTH, 4))            # 把一维阵列array转换为行为image.height、列为image.width的矩阵
#     array = array[:, :, :3]
#     array = array[:, :, ::-1]
#     surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))  # 创建一个与数组上的数据和格式最相似的新Surface。数组可以是2D或3D，具有任意大小的整数值。
#     my_display.blit(surface, (0, 0))
#     image.save_to_disk('/home/piaozx/carla_output/%06d.jpg' % image.frame)
#     return surface

# args
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
# 给绝对路径 记得改位置
parser.add_argument('--save-path',            default='/home/piaozx/文档/carla-code/cube2fisheye/input/', help='Synchronous mode execution')
args = parser.parse_args()


actor_list, sensor_list = [], []
sensor_type = ['rgb']

def main(args):
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    try:
        # 设置同步模式
        original_settings = world.get_settings()
        settings = world.get_settings()

        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()

        # 设置tm
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        
        # 创建小车
        vehicle_bp = blueprint_library.find('vehicle.mercedes-benz.coupe')
        vehicle_bp.set_attribute('color', '0, 0, 0')
        vehicle_transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        vehicle.set_autopilot(True)
        actor_list.append(vehicle)
        # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        # 是否忽略红绿灯
        # tm.ignore_lights_percentage(ego_vehicle, 100)
        # 如果限速30km/h -> 30*(1-10%)=27km/h
        tm.global_percentage_speed_difference(10.0)
        
        # 创建存储路径
        # output_path = '/home/piaozx/文档/carla-code/cube2fisheye/input'
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

        # 添加rgb相机
        sensor_queue = Queue()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        # 设置相机参数
        camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        camera_bp.set_attribute("fov", "90")
        # camera_bp.set_attribute('sensor_tick', '1.0')
        # front
        cam01_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=0))
        cam01 = world.spawn_actor(camera_bp, cam01_transform, attach_to=vehicle)
        cam01.listen(lambda data:sensor_callback(data, sensor_queue, "rgb_front"))
        sensor_list.append(cam01)
        # right
        cam02_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=90))
        cam02 = world.spawn_actor(camera_bp, cam02_transform, attach_to=vehicle)
        cam02.listen(lambda data:sensor_callback(data, sensor_queue, "rgb_right"))
        sensor_list.append(cam02)
        # back
        cam03_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=180))
        cam03 = world.spawn_actor(camera_bp, cam03_transform, attach_to=vehicle)
        cam03.listen(lambda data:sensor_callback(data, sensor_queue, "rgb_back"))
        sensor_list.append(cam03)
        # left
        cam04_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=270))
        cam04 = world.spawn_actor(camera_bp, cam04_transform, attach_to=vehicle)
        cam04.listen(lambda data:sensor_callback(data, sensor_queue, "rgb_left"))
        sensor_list.append(cam04)
        # top
        cam05_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=90))
        cam05 = world.spawn_actor(camera_bp, cam05_transform, attach_to=vehicle)
        cam05.listen(lambda data:sensor_callback(data, sensor_queue, "rgb_top"))
        sensor_list.append(cam05)
        # bottom
        cam06_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=-90))
        cam06 = world.spawn_actor(camera_bp, cam06_transform, attach_to=vehicle)
        cam06.listen(lambda data:sensor_callback(data, sensor_queue, "rgb_bottom"))
        sensor_list.append(cam06)
        # #######################################################
        # === 3, 收集数据 ===
        while True:
            # Tick the server
            world.tick()

            # 设置观察视图
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),carla.Rotation(pitch=-90)))
            
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)

            try:
                rgbs = []
                for i in range (0, len(sensor_list)):
                    s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
                    sensor_type = s_name.split('_')[0]
                    if sensor_type == 'rgb':
                        rgbs.append(_parse_image_cb(s_data))
                
                # 仅用来可视化
                rgb = np.concatenate(rgbs, axis=1)[...,:3]
                cv2.imshow('vizs', visualize_data(rgb))
                cv2.waitKey(100)

            except Empty:
                print("    Some of the sensor information is missed")
        
    finally:
        world.apply_settings(original_settings)
        tm.set_synchronous_mode(False)
        for sensor in sensor_list:
            sensor.destroy()
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!:)")

def mkdir_folder(path):
    for s_type in sensor_type:
        if not os.path.isdir(os.path.join(path, s_type)):
            os.makedirs(os.path.join(path, s_type))
    return True

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))

# modify from world on rail code
def visualize_data(rgb, lidar, imu_yaw, gnss, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)):

    canvas = np.array(rgb[...,::-1])

    # cv2.putText(canvas, f'yaw angle: {imu_yaw:.3f}', (4, 10), *text_args)
    # cv2.putText(canvas, f'log: {gnss[0]:.3f} alt: {gnss[1]:.3f} brake: {gnss[2]:.3f}', (4, 20), *text_args)

    return canvas

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

    