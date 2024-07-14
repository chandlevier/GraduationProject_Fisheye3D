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
import tofisheyex4 as tf
import create_json as js
import monodet3d
import undistort
import evaluate
import cooperation
# random.seed(0)
# 存储数据集的时候，要修改args.save_path、npc.spawn_actors、tf.transform里的save_path和鱼眼图像存储路径
# 以及js.json_script的images_file和create_multijson的mkdir_fold第一项参数

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
parser.add_argument('--save-path',            default='/home/piaozx/文档/carla-code/carlafisheye/dataset/input/', help='Synchronous mode execution')
args = parser.parse_args()

# 图片大小
IM_WIDTH = 1024
IM_HEIGHT = 1024

class boundingbox():
    def __init__(self):
        self.num = ""
        self.cam = ""
        self.ego_loc = ""
        self.frame = ""
        self.x = ""
        self.y = ""
        self.z = ""
        self.l = ""
        self.w = ""
        self.h = ""
        self.yaw = ""
        self.score = ""
        self.bev_coord = ""

def main(args):
    actor_list, sensor_list = [], []
    sensor_type = ['rgb']
    # 创造client
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)   
    world = client.get_world()
    world = client.load_world('Town05_Opt')
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    blueprint_library = world.get_blueprint_library()
    try:
        print("sucessfully TRY")
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        #-------------------------- 初始设置部分 --------------------------#
        # 获取原有模式设置,退出时要还原模式设置
        original_settings = world.get_settings()
        settings = world.get_settings()

        # 设置自动模式
        settings.fixed_delta_seconds = 0.5
        settings.synchronous_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()

        #--------------------------- 无人车部分 ---------------------------#
        # 创建车辆
        ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # 手动规定
        transform_vehicle = carla.Transform(carla.Location(x=-1, y=-15, z=0.6), carla.Rotation(0, -90, 0))
        transform_vehicle2 = carla.Transform(carla.Location(x=-7, y=-25, z=0.6), carla.Rotation(0, 0, 0))
        transform_vehicle3 = carla.Transform(carla.Location(x=4, y=-29.5, z=0.6), carla.Rotation(0, -180, 0))
        # 随机选择
        # transform_vehicle = random.choice(world.get_map().get_spawn_points())
        # transform_vehicle.location += carla.Location(y=-3.5)
        # print("The ego_vehicle is located in %s" % transform_vehicle)
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform_vehicle)
        ego_vehicle2 = world.spawn_actor(ego_vehicle_bp, transform_vehicle2)
        ego_vehicle3 = world.spawn_actor(ego_vehicle_bp, transform_vehicle3)
        actor_list.append(ego_vehicle)
        actor_list.append(ego_vehicle2)
        actor_list.append(ego_vehicle3)
        #--------------------------- 传感器部分 ---------------------------#
        sensor_queue = Queue()  # Create a queue object with a given maximum size.
        cam_bp = blueprint_library.find('sensor.camera.rgb')

        # 设置相机参数
        # cam1:front cam2:right cam3:back cam4:left
        cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        cam_bp.set_attribute("fov", "90")
        # cam_bp.set_attribute('sensor_tick', '0.1')
        # 设置4个鱼眼相机的位置
        cam1_location = carla.Location(x=2.35, z=0.7)
        cam2_location = carla.Location(x=0.65, y=1.1, z=1)
        cam3_location = carla.Location(x=-2.35, z=0.7)
        cam4_location = carla.Location(x=0.65, y=-1.1, z=1)
        # 调用库函数，设置4组相机*
        # camera_script.surfishcams_buildup(world, cam_bp, cam1_location, cam2_location, cam3_location, cam4_location, ego_vehicle, sensor_list, sensor_queue)
        # camera_script.surfishcams_buildup(world, cam_bp, cam1_location, cam2_location, cam3_location, cam4_location, ego_vehicle2, sensor_list, sensor_queue)
        # camera_script.surfishcams_buildup(world, cam_bp, cam1_location, cam2_location, cam3_location, cam4_location, ego_vehicle3, sensor_list, sensor_queue)
        # sensor_list里的顺序: cam4-cam3-cam2-cam1 每个camera:bottom-top-left-right-front
        
        #----------------------- traffic manager 部分 -----------------------#
        # 设置traffic manager
        # 是否忽略红绿灯
        tm.ignore_lights_percentage(ego_vehicle, 100)
        # 如果限速30km/h -> 30*(1+10%)=33km/h
        tm.global_percentage_speed_difference(-50.0)
        # ego_vehicle.set_autopilot(True, tm.get_port())
        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))

        #--------------------------- 其他npc部分 ---------------------------#
        actor_list = npc.spawn_actor_guding(world, blueprint_library, transform_vehicle, actor_list)
        # actor_list = npc.spawn_actors(world, blueprint_library, transform_vehicle, actor_list)
        
        walker1, walker2, actor_list = npc.spawn_walker(world, blueprint_library, transform_vehicle, actor_list)
        control = carla.WalkerControl()        
        #--------------------------- 开始同步运行 ---------------------------#
        n = -1 ########################改改改！
        iouap_list1, meiouap_list1 = [], []
        mATE_list1, mASE_list1, mAOE_list1 = [], [], []
        iouap_list2, meiouap_list2 = [], []
        mATE_list2, mASE_list2, mAOE_list2 = [], [], []
        iouap_list3, meiouap_list3 = [], []
        mATE_list3, mASE_list3, mAOE_list3 = [], [], []
        # 存放协同后的数据
        coiouap_list = []
        mATE_list, mASE_list, mAOE_list = [], [], []
        # 如果只需要显示每帧的PR则用以下格式
        with open("prediction_IOU_evaluation.txt", "w") as file_read:
            file_read.write("frame\tmate\tmase\tmaoe\tiou_ap\tnds\n")
        # with open("merged_IOU_evaluation.txt", "w") as file_read:
        #     file_read.write("frame\tpred_id\tATE\t\tASE\t\tAOE\t\tmethod\n")
        with open("merged_IOU_evaluation1.txt", "w") as file_read:
            file_read.write("frame\tATE\t\tASE\t\tAOE\t\tmAP\t\tNDS\n")
        with open("merged_IOU_evaluation2.txt", "w") as file_read:
            file_read.write("frame\tATE\t\tASE\t\tAOE\t\tmAP\t\tNDS\n")
        with open("merged_IOU_evaluation3.txt", "w") as file_read:
            file_read.write("frame\tATE\t\tASE\t\tAOE\t\tmAP\t\tNDS\n")
        # for i in range(1, 7):
        #     with open("merged_IOU_evaluation"+str(i)+".txt", "w") as file_read:
        #         file_read.write("frame\tpred_id\tATE\t\tASE\t\tAOE\t\tmAP\t\tNDS\n")
        # for i in range(1, 9):
        #     with open("coop_IOU_evaluation"+str(i)+".txt", "w") as file_read:
        #         file_read.write("frame\tATE\t\tASE\t\tAOE\t\tmAP\t\tNDS\t\tmethod\n")
        with open("coop_IOU_evaluation6.txt", "w") as file_read:
            file_read.write("frame\tATE\t\tASE\t\tAOE\t\tmAP\t\tNDS\t\tmethod\n")
        while True:
            # Tick the server
            world.tick()
            # 行人控制代码
            npc.walker_control(walker1, walker2, control)            
            # 让CARLA界面摄像头跟随车动
            loc = ego_vehicle2.get_transform().location
            spectator.set_transform(carla.Transform(carla.Location(x=loc.x,y=loc.y,z=35),
                                                    carla.Rotation(yaw=0,pitch=-90,roll=0)))
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %06d" % w_frame)
            
            # 新建鸟瞰空白图
            image = np.zeros((1024, 1024, 3), np.uint8)
            image.fill(255)
            n += 1
            try:
                rgbs = []
                for i in range (0, len(sensor_list)):   # 在同一帧下遍历4x5个相机，并将获取图像整改为numpy数组添加到rgbs数组中
                    s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                    # print("    Frame: %06d   Sensor: %s" % (s_frame, s_name))
                    sensor_type = s_name.split('_')[0]
                    if sensor_type == 'rgb':
                        rgbs.append(camera_script._parse_image_cb(s_data))
                # 生成鸟瞰真值，记录鸟瞰真值框参数
                gtboundingboxes = []
                ego_loc = [0.0, 0.0, 0.0]
                ego_loc[0] = ego_vehicle.get_transform().location.x
                ego_loc[1] = ego_vehicle.get_transform().location.y
                ego_loc[2] = ego_vehicle.get_transform().rotation.yaw
                ego_loc2 = [0.0, 0.0, 0.0]
                ego_loc2[0] = ego_vehicle2.get_transform().location.x
                ego_loc2[1] = ego_vehicle2.get_transform().location.y
                ego_loc2[2] = ego_vehicle2.get_transform().rotation.yaw
                ego_loc3 = [0.0, 0.0, 0.0]
                ego_loc3[0] = ego_vehicle3.get_transform().location.x
                ego_loc3[1] = ego_vehicle3.get_transform().location.y
                ego_loc3[2] = ego_vehicle3.get_transform().rotation.yaw
                # 每帧更新一次真值框参数gtboundingboxes
                # 对每一个actor绘制鸟瞰真值框，并记录到真值框参数列表gtboundingboxes中
                for i in range(0, len(actor_list)):
                    id = actor_list[i].type_id
                    x = actor_list[i].get_transform().location.x
                    y = actor_list[i].get_transform().location.y
                    z = actor_list[i].get_transform().location.z
                    yaw = actor_list[i].get_transform().rotation.yaw
                    l, w, h = npc.actor_gddimension(world, actor_list[i])
                    # 真值框参数列表不包含ego_vehicle1和2和3？
                    if i > 2:
                        gt_bbox = boundingbox()
                        # num即为真值框的id号, ego_vehicle的id为-2，-1，0
                        gt_bbox.num = i-2
                        gt_bbox.ego_loc = ego_loc
                        gt_bbox.frame = n
                        gt_bbox.x = x * 10
                        gt_bbox.y = z * 10
                        gt_bbox.z = -y * 10
                        gt_bbox.l = l * 10
                        gt_bbox.w = w * 10
                        gt_bbox.h = h * 10
                        gt_bbox.yaw = yaw
                    else:
                        gt_bbox = boundingbox()
                    camera_script.create_bev_groundtruth(image, x, -y, yaw, l, w, w_frame, i, gt_bbox, (180,180,180))
                    if i > 2:
                        gtboundingboxes.append(gt_bbox)
                # # 存储图像*
                # # sensor_list里的顺序: cam4-cam3-cam2-cam1 每个camera:bottom-top-left-right-front
                # camera_script.multirgbcams_store(sensor_list, args, w_frame, rgbs)

                # # args.save_path = "/home/piaozx/文档/carla-code/carlafisheye/dataset/input/"
                # vehicle_list = ["vehicle1/", "vehicle2/", "vehicle3/"]
                # input_file = ["fishf", "fishb", "fishr", "fishl"]
                # for j in range(len(vehicle_list)):
                #     for k in range(len(input_file)):
                #         vehicle = vehicle_list[j]
                #         cam = input_file[k]
                #         #*
                #         input_dir = args.save_path + vehicle + cam
                #         # 将该帧该相机的针孔图像转换成鱼眼图像
                #         tf.transform(input_dir, str(w_frame).zfill(6), n, cam)
                    
                # # 将该帧所有鱼眼图像转换出来后生成该帧图像的标注文件
                # js.json_script(n, 3)
                # 基于该帧鱼眼图像生成矫正图像*
                # print("\nGenerating undistort output...\n")
                # undistort.undistort_script(n)
                # 基于该帧鱼眼图像和矫正图像路径与标注文件路径以及主机车坐标生成检测结果和绝对坐标鸟瞰图
                print("\nGenerating prediction output of fisheye and undistort...\n")
                boundingboxes1, boundingboxes2, boundingboxes3 = monodet3d.multi_detection3d(n, ego_loc, ego_loc2, ego_loc3)

                # 评估代码
                pred_boxes1 = copy.deepcopy(boundingboxes1)
                gt_boxes1 = copy.deepcopy(gtboundingboxes)
                pred_boxes2 = copy.deepcopy(boundingboxes2)
                gt_boxes2 = copy.deepcopy(gtboundingboxes)
                pred_boxes3 = copy.deepcopy(boundingboxes3)
                gt_boxes3 = copy.deepcopy(gtboundingboxes)
                # evaluate.final_evaluate_scripts(gt_boxes1, pred_boxes1, iouap_list1, n)
                # evaluate.final_evaluate_scripts(gt_boxes2, pred_boxes2, iouap_list2, n)
                car1_score1, car1_score2, meboxes1 = evaluate.final_merge_script(gt_boxes1, pred_boxes1, mATE_list1, mASE_list1, mAOE_list1, meiouap_list1, n, "vehicle1")
                car2_score1, car2_score2, meboxes2 = evaluate.final_merge_script(gt_boxes2, pred_boxes2, mATE_list2, mASE_list2, mAOE_list2, meiouap_list2, n, "vehicle2")
                car3_score1, car3_score2, meboxes3 = evaluate.final_merge_script(gt_boxes3, pred_boxes3, mATE_list3, mASE_list3, mAOE_list3, meiouap_list3, n, "vehicle3")

                cooperation.final_coop_bev(gtboundingboxes, meboxes1, meboxes2, meboxes3, ego_loc, ego_loc2, ego_loc3, car1_score2, car2_score2, car3_score2, n, mATE_list, mASE_list, mAOE_list, coiouap_list, w_frame)
                # print(car1_score1, car1_score2, car2_score1, car2_score2)
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
        # print("mAP_IOU = %f\n" % (evaluate.AP_average(iouap_list)))
        # print("maxAP_IOU = %f\n" % (evaluate.find_max(iouap_list)))
        # print("\nfor mATE, mTP1~mTP6 is:", evaluate.mtpall_average(mATE_list))
        # print("\nfor mASE, mTP1~mTP6 is:", evaluate.mtpall_average(mASE_list))
        # print("\nfor mAOE, mTP1~mTP6 is:", evaluate.mtpall_average(mAOE_list))

        mATE1 = evaluate.average(mATE_list1)
        mASE1 = evaluate.average(mASE_list1)
        mAOE1 = evaluate.average(mAOE_list1)
        mAP1 = evaluate.average(iouap_list1)
        merged_mAP1 = evaluate.average(meiouap_list1)
        NDS1 = evaluate.NDS_calculate(merged_mAP1, mATE1, mASE1, mAOE1)
        print("for vehicle1 original prediction result: mAP = %f" % mAP1)
        print("for vehicle1 merged prediction result:\nmATE = %f, mASE = %f, mAOE = %f, mAP = %f, NDS = %f\n" % (mATE1, mASE1, mAOE1, merged_mAP1, NDS1))
        mATE2 = evaluate.average(mATE_list2)
        mASE2 = evaluate.average(mASE_list2)
        mAOE2 = evaluate.average(mAOE_list2)
        mAP2 = evaluate.average(iouap_list2)
        merged_mAP2 = evaluate.average(meiouap_list2)
        NDS2 = evaluate.NDS_calculate(merged_mAP2, mATE2, mASE2, mAOE2)
        print("for vehicle2 original prediction result: mAP = %f" % mAP2)
        print("for vehicle2 merged prediction result:\nmATE = %f, mASE = %f, mAOE = %f, mAP = %f, NDS = %f\n" % (mATE2, mASE2, mAOE2, merged_mAP2, NDS2))
        mATE3 = evaluate.average(mATE_list3)
        mASE3 = evaluate.average(mASE_list3)
        mAOE3 = evaluate.average(mAOE_list3)
        mAP3 = evaluate.average(iouap_list3)
        merged_mAP3 = evaluate.average(meiouap_list3)
        NDS3 = evaluate.NDS_calculate(merged_mAP3, mATE3, mASE3, mAOE3)
        print("for vehicle3 original prediction result: mAP = %f" % mAP3)
        print("for vehicle3 merged prediction result:\nmATE = %f, mASE = %f, mAOE = %f, mAP = %f, NDS = %f\n" % (mATE3, mASE3, mAOE3, merged_mAP3, NDS3))
        
        mATE = evaluate.average(mATE_list)
        mASE = evaluate.average(mASE_list)
        mAOE = evaluate.average(mAOE_list)
        mAP = evaluate.average(coiouap_list)
        NDS = evaluate.NDS_calculate(mAP, mATE, mASE, mAOE)
        print("for cooperation perception result:\nmATE = %f, mASE = %f, mAOE = %f, mAP = %f, NDS = %f\n" % (mATE, mASE, mAOE, mAP, NDS))

        with open("merged_IOU_evaluation1.txt", "a") as file_read:
            file_read.write("for vehicle1 original prediction result: mAP = %f" % mAP1)
            file_read.write("for vehicle1 merged prediction result:\nmATE = %f, mASE = %f, mAOE = %f, mAP = %f, NDS = %f\n" % (mATE1, mASE1, mAOE1, merged_mAP1, NDS1))
        with open("merged_IOU_evaluation2.txt", "a") as file_read:
            file_read.write("for vehicle2 original prediction result: mAP = %f" % mAP2)
            file_read.write("for vehicle2 merged prediction result:\nmATE = %f, mASE = %f, mAOE = %f, mAP = %f, NDS = %f\n" % (mATE2, mASE2, mAOE2, merged_mAP2, NDS2))
        with open("merged_IOU_evaluation3.txt", "a") as file_read:
            file_read.write("for vehicle3 original prediction result: mAP = %f" % mAP3)
            file_read.write("for vehicle3 merged prediction result:\nmATE = %f, mASE = %f, mAOE = %f, mAP = %f, NDS = %f\n" % (mATE3, mASE3, mAOE3, merged_mAP3, NDS3))
        with open("coop_IOU_evaluation6.txt", "a") as file_read:
            file_read.write("for cooperation perception result:\nmATE = %f, mASE = %f, mAOE = %f, mAP = %f, NDS = %f\n" % (mATE, mASE, mAOE, mAP, NDS))
        

if __name__ == "__main__":
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')