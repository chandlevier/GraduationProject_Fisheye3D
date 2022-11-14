#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys


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
     
    time.sleep(5)
    
    
finally:
    for actor in actor_list:
    	actor.destroy()
    print("All cleaned up!:)")
    

