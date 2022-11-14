#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import abc
import glob
import os
import sys
from types import LambdaType
from collections import deque
from collections import namedtuple
 
try:
#输入存放carla环境的路径
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
 
import carla
import random 
import time
import numpy as np
import cv2
import math
 
IM_WIDTH = 800
IM_HEIGHT = 600
SHOW_PREVIEW = True
 
SHOW_CAM = SHOW_PREVIEW
im_width = IM_WIDTH
im_height = IM_HEIGHT

#与服务器建立连接
try:
	client = carla.Client('localhost',2000)
	client.set_timeout(10.0)
	world = client.get_world()
	blueprint_library = world.get_blueprint_library()
	model = blueprint_library.find('vehicle.mercedes-benz.coupe')
	 
	actor_list = []
	transform = random.choice(world.get_map().get_spawn_points()) #spwan_points共265个点，选第一个点作为初始化小车的位置
	vehicle = world.spawn_actor(model , transform)
	actor_list.append(vehicle)
	
	#单独写一个函数（process_img）来处理接收的图像并将其显示在cv图像框中。
	def process_img(image):
	    i = np.array(image.raw_data)
	    i2 = i.reshape((im_height, im_width , 4))
	    i3 = i2[: , : , : 3]
	    if SHOW_CAM:
	    	cv2.imshow("",i3)
	    	cv2.waitKey(1)
 
	cam = blueprint_library.find('sensor.camera.rgb')
	cam.set_attribute('image_size_x',f'{im_width}')
	cam.set_attribute('image_size_y',f'{im_height}')
	cam.set_attribute('fov',f'110')
	 
	#设定传感器的相对位置(x方向偏移2.5，z方向偏移0.7，y方向偏移)
	#需要调整传感器的角度可以在carla.Transform里添加carla.Rotation(roll,pitch,yew),分别代表x,y,z轴
	#不设置角度当前传感器与车头前向保持一直
	transform = carla.Transform(carla.Location(x=2.5 ,z=0.7 ))
	#将传感器附在小车上
	sensor = world.spawn_actor(cam,transform, attach_to=vehicle)
	actor_list.append(sensor)
	#传感器开始监听
	sensor.listen(lambda data: process_img(data))
	
	time.sleep(15)
    
    
finally:
    for actor in actor_list:
    	actor.destroy()
    print("All cleaned up!:)")
    


