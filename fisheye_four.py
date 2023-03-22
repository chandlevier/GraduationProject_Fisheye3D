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

import camera_script
import npc

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

actor_list, sensor_list = [], []
sensor_type = ['rgb']

def main(args):
    # 创造client
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)   
    world = client.get_world()
    world = client.load_world('Town05')
    blueprint_library = world.get_blueprint_library()
    weather = carla.WeatherParameters(
        sun_azimuth_angle = 90,
        sun_altitude_angle = 30
    )
    world.set_weather(weather)
    try:
        # 获取原有模式设置,退出时要还原模式设置
        original_settings = world.get_settings()
        settings = world.get_settings()

        # 设置自动模式
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()
        # weather = world.
        #--------------------------- 无人车部分 ---------------------------#
        # 创建车辆
        ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # 手动规定
        transform_vehicle = carla.Transform(carla.Location(x=-1, y=-15, z=0.6), carla.Rotation(0, -90, 0))
        # 随机选择
        # transform_vehicle = random.choice(world.get_map().get_spawn_points())
        # transform_vehicle.location += carla.Location(y=-3.5)
        # print("The ego_vehicle is located in %s" % transform_vehicle)
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform_vehicle)
        actor_list.append(ego_vehicle)
        
        #--------------------------- 传感器部分 ---------------------------#
        sensor_queue = Queue()  # Create a queue object with a given maximum size.
        cam_bp = blueprint_library.find('sensor.camera.rgb')

        # 设置相机参数
        # cam1:front cam2:right cam3:back cam4:left
        cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        cam_bp.set_attribute("fov", "90")
        # cam_bp.set_attribute('sensor_tick', '0.1')
        cam1_location = carla.Location(x=2.35, z=0.7)
        cam2_location = carla.Location(x=0.65, y=1.1, z=1)
        cam3_location = carla.Location(x=-2.35, z=0.7)
        cam4_location = carla.Location(x=0.65, y=-1.1, z=1)
        # 调用库函数，设置5个相机
        camera_script.rgbcams_buildup(world, cam_bp, cam1_location, ego_vehicle, sensor_list, sensor_queue, yaw0=0)
        camera_script.rgbcams_buildup(world, cam_bp, cam2_location, ego_vehicle, sensor_list, sensor_queue, yaw0=90)
        camera_script.rgbcams_buildup(world, cam_bp, cam3_location, ego_vehicle, sensor_list, sensor_queue, yaw0=180)
        camera_script.rgbcams_buildup(world, cam_bp, cam4_location, ego_vehicle, sensor_list, sensor_queue, yaw0=270)
        # sensor_list里的顺序: cam4-cam3-cam2-cam1 每个camera:bottom-top-left-right-front
        
        #----------------------- traffic manager 部分 -----------------------#
        # 设置traffic manager
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        # 是否忽略红绿灯
        tm.ignore_lights_percentage(ego_vehicle, 100)
        # 如果限速30km/h -> 30*(1+10%)=33km/h
        tm.global_percentage_speed_difference(-50.0)
        # ego_vehicle.set_autopilot(True, tm.get_port())
        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))

        #--------------------------- 其他npc部分 ---------------------------#
        actor_list = npc.spawn_actors(world, blueprint_library, transform_vehicle, actor_list)
        
        walker1, walker2, actor_list = npc.spawn_walker(world, blueprint_library, transform_vehicle, actor_list)
        control = carla.WalkerControl()
        
        for i in range(7):
            print(actor_list[i].type_id)
        #--------------------------- 开始同步运行 ---------------------------#
        while True:
            # Tick the server
            world.tick()

            # 行人控制代码
            npc.walker_control(walker1, walker2, control)
            
            # 让CARLA界面摄像头跟随车动
            loc = ego_vehicle.get_transform().location
            spectator.set_transform(carla.Transform(carla.Location(x=loc.x,y=loc.y,z=35),
                                                    carla.Rotation(yaw=0,pitch=-90,roll=0)))

            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %06d" % w_frame)
            try:
                rgbs = []
                for i in range (0, len(sensor_list)):   # 在同一帧下遍历4x5个相机，并将获取图像整改为numpy数组添加到rgbs数组中
                    s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                    # print("    Frame: %06d   Sensor: %s" % (s_frame, s_name))
                    sensor_type = s_name.split('_')[0]
                    if sensor_type == 'rgb':
                        rgbs.append(camera_script._parse_image_cb(s_data))
                # 存储图像
                # sensor_list里的顺序: cam4-cam3-cam2-cam1 每个camera:bottom-top-left-right-front
                camera_script.rgbcams_store(sensor_list, args, w_frame, rgbs)
                # 可视化 单纯把几个相机的图像拼接起来展示
                # rgb = np.concatenate(rgbs, axis=1)[...,:3]
                # cv2.imshow('rgb', camera_script.visualize_data(rgb))
                # cv2.waitKey(100)                
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


if __name__ == "__main__":
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')