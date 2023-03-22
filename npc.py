import carla
import random


def spawn_actors(world, world_bp_lib, transform_vehicle, actor_list):
    # 添加其他actor
    transform_actor = random.choice(world.get_map().get_spawn_points())
    # actor1 右前方右四停车位要开出来的动态车辆
    actor1_bp = world_bp_lib.find('vehicle.mercedes-benz.coupe')
    transform_actor.location = transform_vehicle.location + carla.Location(x=5, y=-12)
    transform_actor.rotation.yaw = -180.0
    actor1 = world.try_spawn_actor(actor1_bp, transform_actor)
    if actor1 is not None:
        actor_list.append(actor1)
        actor1.set_autopilot(True)
        print('created %s' % actor1.type_id)

    # actor2 左前方左一停车位要开出来的动态车辆
    actor2_bp = world_bp_lib.find('vehicle.tesla.model3')
    transform_actor.location = transform_vehicle.location + carla.Location(x=-6, y=-4)
    transform_actor.rotation.yaw = 0
    actor2 = world.try_spawn_actor(actor2_bp, transform_actor)
    if actor2 is not None:
        actor_list.append(actor2)
        actor2.set_autopilot(True)
        print('created %s' % actor2.type_id)

    # actor3 右前方右二停车位的静止车
    actor3_bp = world_bp_lib.find('vehicle.audi.a2')
    transform_actor.location = transform_vehicle.location + carla.Location(x=5, y=-6.5)
    transform_actor.rotation.yaw = 0
    actor3 = world.try_spawn_actor(actor3_bp, transform_actor)
    if actor3 is not None:
        actor_list.append(actor3)
        print('created %s' % actor3.type_id)

    # actor4 右前方右一停车位的静止车
    actor4_bp = world_bp_lib.find('vehicle.bmw.grandtourer')
    transform_actor.location = transform_vehicle.location + carla.Location(x=5, y=-4)
    transform_actor.rotation.yaw = -180
    actor4 = world.try_spawn_actor(actor4_bp, transform_actor)
    if actor4 is not None:
        actor_list.append(actor4)
        print('created %s' % actor4.type_id)
    
    return actor_list

def spawn_walker(world, world_bp_lib, transform_vehicle, actor_list):
    walker_bp = random.choice(world_bp_lib.filter('walker.pedestrian.*'))
    transform_walker = random.choice(world.get_map().get_spawn_points())
    transform_walker.location = transform_vehicle.location + carla.Location(x=6, y=-8)
    walker1 = world.try_spawn_actor(walker_bp, transform_walker)
    transform_walker.location = transform_vehicle.location + carla.Location(x=-8, y=-11)
    walker2 = world.try_spawn_actor(walker_bp, transform_walker)
    if walker1 is not None:
        actor_list.append(walker1)
        print('created %s' % walker1.type_id)
    if walker2 is not None:
        actor_list.append(walker2)
        print('created %s' % walker2.type_id)
    return walker1, walker2, actor_list

def walker_control(walker1, walker2, control):
    revert_flag = False
    control.direction.y = 0
    control.direction.z = 0
    control1, control2 = control, control
    control1.speed = 3
    if(walker1.get_location().x<-10):
        revert_flag = True
    elif(walker1.get_location().x>6):
        revert_flag = False
    
    if(revert_flag):
        control1.direction.x = 1
    else:
        control1.direction.x = -1
    walker1.apply_control(control1)
    control2.speed = 2
    control2.direction.x = 1
    walker2.apply_control(control)


# def spawn_walkers(args, world, client, walkers_list, logging, all_id):
#     blueprintsWalkers = world.get_blueprint_library().filter(args.filterw) # Return一个可用行人角色的bp列表，方便在世界中生成。carla提供了26种walker bp
#     SpawnActor = carla.command.SpawnActor

#     percentagePedestriansRunning = 0.0      # 设置变量，表示how many pedestrians will run
#     percentagePedestriansCrossing = 0.0     # 设置变量，表示how many pedestrians will walk through the road
#     # 1. take all the random locations to spawn
#     spawn_points = []
#     for i in range(args.number_of_walkers): # 先生成walkers数量的生成点
#         spawn_point = carla.Transform()
#         loc = world.get_random_location_from_navigation() # 生成walker行走路线的目的地点。carla.WalkerAIController会使用函数go_to_location()控制行人走向该点
#         if (loc != None):
#             spawn_point.location = loc      # 将获取到的随机目的地点的坐标赋给生成点，并将该生成点放入生成点列表中
#             spawn_points.append(spawn_point)
#     # 2. we spawn the walker object
#     batch = []
#     walker_speed = []
#     for spawn_point in spawn_points:
#         walker_bp = random.choice(blueprintsWalkers)      # 从最开始获取到的可用行人蓝图列表中随机抽取单个蓝图，作为当前循环的行人蓝图
#         # set as not invincible 将该行人的属性设置为not invincible
#         if walker_bp.has_attribute('is_invincible'):
#             walker_bp.set_attribute('is_invincible', 'false')
#         # set the max speed 设置行人的速度属性
#         if walker_bp.has_attribute('speed'):
#             if (random.random() > percentagePedestriansRunning): # random.random()返回半开区间[0.0,1.0)内的随机浮点数
#                 # walking
#                 walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1]) # 随机让当前行人处于走路的速度，并将该速度加入到walker_speed列表中
#             else:
#                 # running
#                 walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2]) # 随机让当前行人处于跑步的速度，并将该速度加入到walker_speed列表中
#         else:   # walkers都有speed属性，不明白这个else是为了什么，是以防万一吗？
#             print("Walker has no speed")
#             walker_speed.append(0.0)
#         batch.append(SpawnActor(walker_bp, spawn_point))# 本次循环的最后，将该循环中确定好bp、speed、生成点的行人进行生成，并加入到batch列表中
#     results = client.apply_batch_sync(batch, True)      # 执行batch命令，并在执行完毕后执行一次world.tick
#     walker_speed2 = []
#     for i in range(len(results)):
#         if results[i].error:
#             logging.error(results[i].error)
#         else:
#             walkers_list.append({"id": results[i].actor_id})
#             walker_speed2.append(walker_speed[i])
#     walker_speed = walker_speed2
#     # 3. we spawn the walker controller
#     batch = []
#     walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
#     for i in range(len(walkers_list)):
#         batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))    # 为每个行人生成controller
#     results = client.apply_batch_sync(batch, True)
#     for i in range(len(results)):           # 若生成失败，则报错；否则将controller天加入walkers_list的con属性中
#         if results[i].error:
#             logging.error(results[i].error)
#         else:
#             walkers_list[i]["con"] = results[i].actor_id
#     # 4. we put altogether the walkers and controllers id to get the objects from their id
#     for i in range(len(walkers_list)):
#         all_id.append(walkers_list[i]["con"]) # 把walkers_list中的id属性和con属性都放入all_id列表中，并靠此列表获取所有的actor
#         all_id.append(walkers_list[i]["id"])
#     all_actors = world.get_actors(all_id)

#     # wait for a tick to ensure client receives the last transform of the walkers we have just created
#     world.tick()

#     # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
#     # set how many pedestrians can cross the road
#     world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
#     for i in range(0, len(all_id), 2):
#         # start walker
#         all_actors[i].start()
#         # set walk to random point
#         all_actors[i].go_to_location(world.get_random_location_from_navigation())
#         # max speed
#         all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))