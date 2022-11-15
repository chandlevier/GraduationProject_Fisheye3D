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
import cv2

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
    i = np.array(image.raw_data)
#    print(dir(i))
#    print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    #reshape:改变数组或矩阵的形状,为一个IM_HEIGHT行的IM_WIDTH维新数组
    i3 = i2[:, :, :3]
#    print(i3)
    cv2.imshow("video",i3)
    cv2.waitKey(5000)
    print("get an image!")
    return i3/255.0

#def process_image(image):
#    print("get a img!")
#    image.save_to_disk('out/%06d.png' % image.frame)
#    i = np.array(image.raw_data)  # convert to an array
#    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # was flattened, so we're going to shape it.
#    i3 = i2[:, :, :3]  # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
#    return i3/255.0  # normalize


actor_list = []

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    
    world = client.get_world()
    
    blueprint_library = world.get_blueprint_library()
    
    vehicle_bp = blueprint_library.find('vehicle.mercedes-benz.coupe')
    vehicle_bp.set_attribute('color', '0, 0, 0')
    print(vehicle_bp)
    
    spawn_point = random.choice(world.get_map().get_spawn_points())
    
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    #vehicle.set_autopilot(True)
    
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
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
    
    #spectator
    while True:
	    spectator = world.get_spectator()
	    transform = vehicle.get_transform()
	    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),carla.Rotation(pitch=-90)))
	    time.sleep(0.005)
    
    actor_list.append(camera)
    
    
    time.sleep(50)
    
    
finally:
    for actor in actor_list:
    	actor.destroy()
    print("All cleaned up!:)")
    

