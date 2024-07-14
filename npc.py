import carla
import random

# def scale_confirm(id):
#     if id[0] == 'v':
#         if id[8:10]=="bh" or id[8:14]=="yamaha" or id[8:15]=="gazelle" or id[8:16]=="kawasaki" or id[8:19]=="diamondback" or id[8:14]=="harley":    # 生成骑摩托车自行车的actor的尺寸
#             w = 1.8
#             l = 2
#         elif id[8:12]=="mini":
#             w = 1.68
#             l = 3.7
#         elif id[8:24]=="tesla.cybertruck":
#             w, l = 2, 5.8
#         else:
#             w, l = 1.8, 4.7
#     elif id[0] == 'w':
#         w = 1
#         l = 1
#     return l, w

def actor_gddimension(world, actor):
    vehicle_dimensions = actor.bounding_box
    length = vehicle_dimensions.extent.x * 2
    width = vehicle_dimensions.extent.y * 2
    height = vehicle_dimensions.extent.z * 2
    return length, width, height


def spawn_actor_guding(world, world_bp_lib, transform_vehicle, actor_list):
    # 添加其他actor
    transform_actor = random.choice(world.get_map().get_spawn_points())

    delta_x = [5, -6]
    delta_y = -2.8
    # -------------------------- 车辆actor固定生成部分 -------------------------- #
    # actor1 右前方右一停车位的静止车
    actor1_bp = world_bp_lib.find('vehicle.bmw.grandtourer')
    transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[0], y=-4)
    transform_actor.rotation.yaw = -180
    actor1 = world.try_spawn_actor(actor1_bp, transform_actor)
    if actor1 is not None:
        actor_list.append(actor1)
        print('created %s' % actor1.type_id)
    # actor2 右前方右二停车位的静止车
    actor2_bp = world_bp_lib.find('vehicle.audi.a2')
    transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[0], y=-6.5)
    transform_actor.rotation.yaw = 0
    actor2 = world.try_spawn_actor(actor2_bp, transform_actor)
    if actor2 is not None:
        actor_list.append(actor2)
        print('created %s' % actor2.type_id)
    # actor3 右前方右三停车位的静止车
    actor3_bp = world_bp_lib.find('vehicle.audi.etron')
    transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[0], y=-9.5)
    transform_actor.rotation.yaw = 0
    actor3 = world.try_spawn_actor(actor3_bp, transform_actor)
    if actor3 is not None:
        actor_list.append(actor3)
        # actor4.set_autopilot(True)
        print('created %s' % actor3.type_id)
    # actor4 右前方右四停车位要开出来的动态车辆
    actor4_bp = world_bp_lib.find('vehicle.bh.crossbike')
    transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[0], y=-12)
    transform_actor.rotation.yaw = -180.0
    actor4 = world.try_spawn_actor(actor4_bp, transform_actor)
    if actor4 is not None:
        actor_list.append(actor4)
        # actor4.set_autopilot(True)
        print('created %s' % actor4.type_id)
    # # actor5 右前方右五停车位要开出来的动态车辆
    # actor5_bp = random.choice(world_bp_lib.filter('vehicle.tesla.model3'))
    # transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[0], y=-14.5)
    # transform_actor.rotation.yaw = 0
    # actor5 = world.try_spawn_actor(actor5_bp, transform_actor)
    # if actor5 is not None:
    #     actor_list.append(actor5)
    #     # actor5.set_autopilot(True)
    #     print('created %s' % actor5.type_id)
    # actor5 右前方左六停车位
    actor5_bp = random.choice(world_bp_lib.filter('vehicle.tesla.model3'))
    transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[1], y=-17.5)
    transform_actor.rotation.yaw = 0
    actor5 = world.try_spawn_actor(actor5_bp, transform_actor)
    if actor5 is not None:
        actor_list.append(actor5)
        # actor5.set_autopilot(True)
        print('created %s' % actor5.type_id)
    # actor6 右前方右六停车位要开出来的动态车辆
    actor6_bp = world_bp_lib.find('vehicle.tesla.cybertruck')
    transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[0], y=-17.5)
    transform_actor.rotation.yaw = 0
    actor6 = world.try_spawn_actor(actor6_bp, transform_actor)
    if actor6 is not None:
        actor_list.append(actor6)
        # actor6.set_autopilot(True)
        print('created %s' % actor6.type_id)
    # actor7 左前方左一停车位要开出来的动态车辆
    actor7_bp = world_bp_lib.find('vehicle.tesla.model3')
    transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[1], y=-4)
    transform_actor.rotation.yaw = 0
    actor7 = world.try_spawn_actor(actor7_bp, transform_actor)
    if actor7 is not None:
        actor_list.append(actor7)
        # actor7.set_autopilot(True)
        print('created %s' % actor7.type_id)
    # actor8 左前方左二停车位要开出来的动态车辆
    actor8_bp = world_bp_lib.find('vehicle.carlamotors.carlacola')
    transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[1], y=-7)
    transform_actor.rotation.yaw = 0
    actor8 = world.try_spawn_actor(actor8_bp, transform_actor)
    if actor8 is not None:
        actor_list.append(actor8)
        # actor4.set_autopilot(True)
        print('created %s' % actor8.type_id)
    
    # actor10 左前方左四停车位要开出来的动态车辆
    actor10_bp = world_bp_lib.find('vehicle.yamaha.yzf')
    transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[1], y=-12.5)
    transform_actor.rotation.yaw = 0
    actor10 = world.try_spawn_actor(actor10_bp, transform_actor)
    if actor10 is not None:
        actor_list.append(actor10)
        # actor4.set_autopilot(True)
        print('created %s' % actor10.type_id)
    return actor_list

def spawn_actors(world, world_bp_lib, transform_vehicle, actor_list):
    # 添加其他actor
    transform_actor = random.choice(world.get_map().get_spawn_points())

    delta_x = [5, -6]
    delta_y = -2.8
        # -------------------------- 车辆actor随机生成部分 -------------------------- #
    yaw = [0, 180]
    # for i in range(random.randint(14,20)):
    #     actor_bp = random.choice(world_bp_lib.filter('vehicle'))
    #     transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[random.randint(0,1)], y=-4+delta_y*random.randint(0,9))
    #     transform_actor.rotation.yaw = yaw[random.randint(0,1)]
    #     actor = world.try_spawn_actor(actor_bp, transform_actor)
    #     if actor is not None:
    #         actor_list.append(actor)
    #         print('created %s' % actor.type_id)
    for i in range(random.randint(2,4)):
        actor_bp = random.choice(world_bp_lib.filter('vehicle'))
        transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[0], y=-4+delta_y*random.randint(0,4))
        transform_actor.rotation.yaw = yaw[random.randint(0,1)]
        actor = world.try_spawn_actor(actor_bp, transform_actor)
        if actor is not None:
            actor_list.append(actor)
            print('created %s' % actor.type_id)
    for i in range(random.randint(4,6)):
        actor_bp = random.choice(world_bp_lib.filter('vehicle'))
        transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[0], y=-4+delta_y*random.randint(5,10))
        transform_actor.rotation.yaw = yaw[random.randint(0,1)]
        actor = world.try_spawn_actor(actor_bp, transform_actor)
        if actor is not None:
            actor_list.append(actor)
            print('created %s' % actor.type_id)
    for i in range(random.randint(0,2)):
        actor_bp = random.choice(world_bp_lib.filter('vehicle'))
        transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[1], y=-4+delta_y*random.randint(0,1))
        transform_actor.rotation.yaw = yaw[random.randint(0,1)]
        actor = world.try_spawn_actor(actor_bp, transform_actor)
        if actor is not None:
            actor_list.append(actor)
            print('created %s' % actor.type_id)
    for i in range(random.randint(4,6)):
        actor_bp = random.choice(world_bp_lib.filter('vehicle'))
        transform_actor.location = transform_vehicle.location + carla.Location(x=delta_x[1], y=-4+delta_y*random.randint(3,9))
        transform_actor.rotation.yaw = yaw[random.randint(0,1)]
        actor = world.try_spawn_actor(actor_bp, transform_actor)
        if actor is not None:
            actor_list.append(actor)
            print('created %s' % actor.type_id)

    
    return actor_list

def spawn_walker(world, world_bp_lib, transform_vehicle, actor_list):
    # walker_bp = random.choice(world_bp_lib.filter('walker'))
    walker1_bp = world_bp_lib.find("walker.pedestrian.0001")
    walker2_bp = world_bp_lib.find("walker.pedestrian.0002")
    transform_walker = random.choice(world.get_map().get_spawn_points())
    transform_walker.location = transform_vehicle.location + carla.Location(x=6, y=-8)
    walker1 = world.try_spawn_actor(walker1_bp, transform_walker)
    transform_walker.location = transform_vehicle.location + carla.Location(x=-8, y=-11)
    walker2 = world.try_spawn_actor(walker2_bp, transform_walker)
    if walker1 is not None:
        actor_list.append(walker1)
        print('created walker1 %s' % walker1.type_id)
    if walker2 is not None:
        actor_list.append(walker2)
        print('created walker2 %s' % walker2.type_id)
    control = carla.WalkerControl()
    # for i in range(random.randint(1,4)):
    #     transform = random.choice(world.get_map().get_spawn_points())
    #     transform.location = transform_vehicle.location + carla.Location(x=random.randint(-10,10), y=-random.randint(5,30))
    #     walker_bp = random.choice(world_bp_lib.filter('walker'))
        
    #     walker = world.try_spawn_actor(walker_bp, transform)
    #     control.direction.x = random.uniform(-1, 1)
    #     control.direction.y = random.uniform(-1, 1)
    #     control.direction.z = random.uniform(-1, 1)
    #     control.speed = random.uniform(1,5)
    #     if walker is not None:
    #         actor_list.append(walker)
    #         walker.apply_control(control)
    #         print('created walker_npc %s' % walker.type_id)

    return walker1, walker2, actor_list

def walker_control(walker1, walker2, control):
    revert_flag = False
    control.direction.y = 0
    control.direction.z = 0
    control1, control2 = control, control
    control1.speed = 3
    if walker1.get_location().x < -10:
        revert_flag = True
    elif walker1.get_location().x > 6:
        revert_flag = False
    
    if(revert_flag):
        control1.direction.x = 1
    else:
        control1.direction.x = -1
    walker1.apply_control(control1)
    control2.speed = 2
    control2.direction.x = 1
    walker2.apply_control(control)
