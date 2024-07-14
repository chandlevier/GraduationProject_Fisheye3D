import math
from mmdet3d.core.visualizer import image_vis
import numpy as np
import cv2
import camera_script
import copy
import evaluate

def IOU_coop(global_box, coop_method="coop1"):
    # 隔离
    boxes = copy.deepcopy(global_box)
    final_box = []
    # 标志清空为ture
    for i in range(len(boxes)):
        boxes[i].flag = "True"
    if len(boxes) == 1:
        final_box.append(boxes[0])
    else:
        # 确定是否是非极大值抑制法
        if coop_method == "coop1":
            return coop1(boxes)
        else:
            for i in range(0, len(boxes)-1):
                # 确认当前检测框没有被融合过，否则直接跳过
                if boxes[i].flag == "True":
                    for j in range(1, len(boxes)):
                        if boxes[i].pred_id == boxes[j].pred_id:
                            if coop_method == "coop2":
                                boxes[i] == coop2(boxes[i], boxes[j])
                            elif coop_method == "coop3":
                                boxes[i] == coop3(boxes[i], boxes[j])
                            elif coop_method == "coop4":
                                boxes[i] == coop4(boxes[i], boxes[j])
                            boxes[j].flag == "False"
                    boxes[i].flag == "False"
                    final_box.append(boxes[i])
    if len(boxes) != 0:
        if boxes[-1].flag == "True":
            final_box.append(boxes[-1])
    
    return final_box

# 计算协同后的检测框的mTP评估函数
def coIOU_mTPeval(gt_boxes, pred_bboxes, frame, score_method="score1", coop_method="coop1"):
    bboxes = copy.deepcopy(pred_bboxes)
    ATE, ASE, AOE = 0, 0, 0
    for i in range(len(pred_bboxes)):
        id = pred_bboxes[i].pred_id - 1
        iou = prepare_iou(pred_bboxes[i], gt_boxes[id])
        pred_bboxes[i].iou = iou
    for i in range(len(bboxes)):
        # 平均位移误差ATE：目标中心在二维平面上的欧式距离
        id = bboxes[i].pred_id - 1
        ate = pow(pow((bboxes[i].x-gt_boxes[id].x),2)+pow((bboxes[i].z-gt_boxes[id].z),2), 0.5)
        # 平均尺度误差ASE：预测框与真值框的1-iou
        ase = 1 - bboxes[i].iou
        # 平均方向误差AOE：预测框与帧值框的最小偏航角度误差
        aoe = bboxes[i].yaw - gt_boxes[id].yaw
        ATE += ate
        ASE += ase
        AOE += aoe
        with open("merged_IOU_evaluation.txt", "a") as file_read:
            file_read.write("%i\t%i\t\t%f\t%f\t%f\t%s\n" % (frame, id+1, ate, ase, aoe, score_method+coop_method))
    if len(bboxes) != 0:
        return ATE/len(bboxes), ASE/len(bboxes), AOE/len(bboxes)
    else:
        return 0, 0, 0
    
# 协同后的检测框的PR评估函数
def coIOU_evaluate(gt_bboxes, pred_bboxes, iou_thres, frame, prec, rec, score_thres, score_method="score1", coop_method="coop1"):
    bboxes = copy.deepcopy(pred_bboxes)
    TP, FP = 0, 0
    # 计算融合后的pr
    for i in range(len(bboxes)):
        if bboxes[i].score >= score_thres and bboxes[i].iou >= iou_thres:   TP += 1
        elif bboxes[i].score >= score_thres and bboxes[i].iou < iou_thres:  FP += 1
    GT = gtrange_calculate(gt_bboxes)
    FN = GT - TP - FP
    if TP == 0 and FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    # with open("merged_IOU_evaluation.txt", "a") as file_read:
    #     file_read.write("%i\t%f\t%f\t%f\t%s\n" % (frame, score_thres, precision, recall, score_method+merge_method))
    prec.append(precision)
    rec.append(recall)


def coop_script(gt_boxes, pred_boxes1, pred_boxes2, ego_loc1, ego_loc2, car1_score1, car1_score2, car2_score1, car2_score2, frame, mATE_list, mASE_list, mAOE_list, coiouap_list):
    # 隔离处理列表和原单车融合结果列表
    prec, rec = [], []
    boxes1 = copy.deepcopy(pred_boxes1)
    boxes2 = copy.deepcopy(pred_boxes2)
    coop_method = ["coop1", "coop2", "coop3", "coop4"]
    score_list = []
    for i in range(0,len(coop_method)):
        for j in range(2):
            # 第一种车辆检测评分方法——mtp
            if j == 0:
                boxes1 = car_score(boxes1, car1_score1)
                boxes2 = car_score(boxes2, car2_score1)
                # 增加一个全局目标列表供协同用
                box = copy.deepcopy(boxes1)
                for k in range(len(boxes2)):
                    box.append(boxes2[k])
                coop_box = IOU_coop(box, coop_method[i])
                evaluate.meIOU_calculate(gt_boxes, coop_box)
                mate, mase, maoe = coIOU_mTPeval(gt_boxes, coop_box, frame, "score1", coop_method[i])
                mATE_list.append(mate)
                mASE_list.append(mase)
                mAOE_list.append(maoe)
                score_list = evaluate.score_sort(score_list, coop_box)
                for score in score_list:
                    coIOU_evaluate(gt_boxes, coop_box, 0.3, frame, prec, rec, score, "score1", coop_method[i])
                coiou_ap = voc_ap(rec, prec)
                coiouap_list.append(coiou_ap)
                NDS = NDS_calculate(coiou_ap, mate, mase, maoe)
                read_in_file(i, j, frame, mate, mase, maoe, coiou_ap, NDS)
                coopbev_printer(coop_box, frame, "score1", coop_method[i], ego_loc1, ego_loc2)
            # 第二种车辆检测评分方法——NDS
            elif j ==1 :
                boxes1 = car_score(boxes1, car1_score2)
                boxes2 = car_score(boxes2, car2_score2)
                # 增加一个全局目标列表供协同用
                box = copy.deepcopy(boxes1)
                for k in range(len(boxes2)):
                    box.append(boxes2[k])
                coop_box = IOU_coop(box, coop_method[i])
                evaluate.meIOU_calculate(gt_boxes, coop_box)
                mate, mase, maoe = coIOU_mTPeval(gt_boxes, coop_box, frame, "score2", coop_method[i])
                mATE_list.append(mate)
                mASE_list.append(mase)
                mAOE_list.append(maoe)
                score_list = evaluate.score_sort(score_list, coop_box)
                for score in score_list:
                    coIOU_evaluate(gt_boxes, coop_box, 0.3, frame, prec, rec, score, "score2", coop_method[i])
                coiou_ap = voc_ap(rec, prec)
                coiouap_list.append(coiou_ap)
                NDS = NDS_calculate(coiou_ap, mate, mase, maoe)
                read_in_file(i, j, frame, mate, mase, maoe, coiou_ap, NDS)
                coopbev_printer(coop_box, frame, "score2", coop_method[i], ego_loc1, ego_loc2)

def final_coop_bev(gt_boxes, pred_boxes1, pred_boxes2, pred_boxes3, ego_loc1, ego_loc2, ego_loc3, car1_score2, car2_score2, car3_score2, frame, mATE_list, mASE_list, mAOE_list, coiouap_list, w_frame):
    # 隔离处理列表和原单车融合结果列表
    prec, rec = [], []
    boxes1 = copy.deepcopy(pred_boxes1)
    boxes2 = copy.deepcopy(pred_boxes2)
    boxes3 = copy.deepcopy(pred_boxes3)
    score_list = []
    # 第二种车辆检测评分方法——NDS
    boxes1 = car_score(boxes1, car1_score2)
    boxes2 = car_score(boxes2, car2_score2)
    boxes3 = car_score(boxes3, car3_score2)
    # 增加一个全局目标列表供协同用
    box = copy.deepcopy(boxes1)
    for k in range(len(boxes2)):
        box.append(boxes2[k])
    for l in range(len(boxes3)):
        box.append(boxes3[l])
    # 加权融合法
    coop_box = IOU_coop(box, "coop3")
    evaluate.meIOU_calculate(gt_boxes, coop_box)
    mate, mase, maoe = coIOU_mTPeval(gt_boxes, coop_box, frame, "score2", "coop3")
    mATE_list.append(mate)
    mASE_list.append(mase)
    mAOE_list.append(maoe)
    score_list = evaluate.score_sort(score_list, coop_box)
    for score in score_list:
        coIOU_evaluate(gt_boxes, coop_box, 0.3, frame, prec, rec, score, "score2", "coop3")
    coiou_ap = voc_ap(rec, prec)
    coiouap_list.append(coiou_ap)
    NDS = NDS_calculate(coiou_ap, mate, mase, maoe)
    read_in_file(2, 1, frame, mate, mase, maoe, coiou_ap, NDS)
    coopbev_printer(coop_box, frame, "score2", "coop3", ego_loc1, ego_loc2, ego_loc3, w_frame)



def coopbev_printer(coop_boxes, frame, score_method, coop_method, ego_loc1, ego_loc2, ego_loc3, w_frame):
    # 画图
    image = np.zeros((1024, 1024, 3), np.uint8)
    image.fill(255)
    font = cv2.FONT_HERSHEY_COMPLEX
    image_vis.create_egobev(image, ego_loc1, "cam", font,(2,204,244))
    image_vis.create_egobev(image, ego_loc2, "cam", font,(2,204,244))
    image_vis.create_egobev(image, ego_loc3, "cam", font,(2,204,244))
    for i in range(len(coop_boxes)):
        u1, u2, u3, u4, v1, v2, v3, v4 = tuple(coop_boxes[i].bev_coord)
        image_vis.bev_printer(image, int(u1), int(u2), int(u3), int(u4), int(v1), int(v2), int(v3), int(v4), font, coop_boxes[i].num+1, coop_boxes[i].score, (2,204,244))
    camera_script.mkdir_folder('/home/piaozx/文档/carla-code/carlafisheye/', "coop_bev", score_method+coop_method)
    bev_filename = "/home/piaozx/文档/carla-code/carlafisheye/coop_bev/" + score_method+coop_method + "/frame" + str(frame) + "_BEV.png"
    cv2.imwrite(bev_filename, image)

    img_ground = cv2.imread("bev_groundtruth/frame" + str(w_frame).zfill(6) + ".png")
    miximg = np.zeros((1024, 1024, 3), np.uint8)
    miximg = cv2.addWeighted(image, 0.5, img_ground, 0.5, 0)
    cv2.imshow("cooperative perception bev", miximg)
    cv2.waitKey(50)

# -------------------------------- 以下函数为不用再看的数据计算函数 -------------------------------- #

def NDS_calculate(map, mate, mase, maoe):
    return (7*map + (1-min(1,mate/10)) + (1-min(1,mase)) + (1-min(1,maoe)))/10


def average(list):
    sum = 0
    if len(list) != 0:
        for i in range(len(list)):
            sum += list[i]
        return sum/len(list)
    else:
        return 0

def mtpall_average(mtp_list):
    mtp1_sum, mtp2_sum, mtp3_sum, mtp4_sum, mtp5_sum, mtp6_sum = 0, 0, 0, 0, 0, 0
    if len(mtp_list) != 0:
        for i in range(0, len(mtp_list), 6):
            mtp1_sum += mtp_list[i]
            mtp2_sum += mtp_list[i+1]
            mtp3_sum += mtp_list[i+2]
            mtp4_sum += mtp_list[i+3]
            mtp5_sum += mtp_list[i+4]
            mtp6_sum += mtp_list[i+5]
        return mtp1_sum/(len(mtp_list)/6), mtp2_sum/(len(mtp_list)/6), mtp3_sum/(len(mtp_list)/6), mtp4_sum/(len(mtp_list)/6), mtp5_sum/(len(mtp_list)/6), mtp6_sum/(len(mtp_list)/6)
    else:
        return 0

def AP_average(ap_list):
    AP1_sum, AP2_sum, AP3_sum, AP4_sum = 0, 0, 0, 0
    if len(ap_list) != 0:
        for i in range(0, len(ap_list), 4):
            AP1_sum += ap_list[i]
            AP2_sum += ap_list[i+1]
            AP3_sum += ap_list[i+2]
            AP4_sum += ap_list[i+3]
        return AP1_sum/(len(ap_list)/4), AP2_sum/(len(ap_list)/4), AP3_sum/(len(ap_list)/4), AP4_sum/(len(ap_list)/4)
    else:
        return 0

def score_sort(score_list, boxes_list):
    for i in range(len(boxes_list)):
        score_list.append(boxes_list[i].score)
    score_list = list(set(score_list))
    score_list.sort(reverse = True)
    return score_list

# 融合方法x4
# 第1种：非极大值抑制法
def coop1(global_boxes):
    boxes = copy.deepcopy(global_boxes)
    box_list = []
    final_box = []
    # 将所有检测框的左上角坐标、右下角坐标与置信度放进一个以数组为元素的数组中
    for i in range(len(boxes)):
        box=[]
        minnum, maxnum = find_minandmax(boxes[i].bev_coord)
        box.append(boxes[i].bev_coord[minnum])
        box.append(boxes[i].bev_coord[minnum+4])
        box.append(boxes[i].bev_coord[maxnum])
        box.append(boxes[i].bev_coord[maxnum+4])
        box.append(float(boxes[i].score))
        box_list.append(box)
        # np.append(box_list, box)
    
    index = py_cpu_nms(box_list)
    for i in index:
        final_box.append(boxes[i])
    return final_box

def py_cpu_nms(boxes, thresh=0):
    dets = np.array(copy.deepcopy(boxes))
    # x1, y1, x2, y2, scores = [], [], [], [], []
    # for i in range(len(dets)):
    #     x1.append(dets[i][0])
    #     y1.append(dets[i][1])
    #     x2.append(dets[i][2])
    #     y2.append(dets[i][3])
    #     scores.append(dets[i][4])
    # 边界框的坐标
    x1 = dets[:, 0] # 所有行第一列
    y1 = dets[:, 1] # 所有行第二列
    x2 = dets[:, 2] # 所有行第三列
    y2 = dets[:, 3] # 所有行第四列
    # 计算边界框的面积
    areas = (y2 - y1 + 1) * (x2 - x1 + 1) # (第四列 - 第二列 + 1) * (第三列 - 第一列 + 1)
    # 提取置信度
    scores = dets[:, 4] # 所有行第五列
    keep = []   # 保留
 
    # 按边界框的置信度得分排序   尾部加上[::-1] 倒序的意思 如果没有[::-1] argsort返回的是从小到大的
    index = scores.argsort()[::-1]  # 对所有行的第五列进行从大到小排序，返回索引值
 
    #迭代边界框
    while index.size > 0: # 6 > 0,      3 > 0,      2 > 0
        i = index[0]  # every time the first is the biggst, and add it directly每次第一个是最大的，直接加进去
        keep.append(i)#保存
        #计算并集上交点的纵坐标（IOU）
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap计算重叠点
        y11 = np.maximum(y1[i], y1[index[1:]])  # index[1:] 从下标为1的数开始，直到结束
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
 
        #计算并集上的相交面积
        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap重叠权值、宽度
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap重叠高度
        overlaps = w * h# 重叠部分、交集
 
        #IoU：intersection-over-union的本质是搜索局部极大值，抑制非极大值元素。即两个边界框的交集部分除以它们的并集。
        #          重叠部分 / （面积[i] + 面积[索引[1:]] - 重叠部分）
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)#重叠部分就是交集，iou = 交集 / 并集
        # print("ious", ious)
        #               ious <= 0.7
        idx = np.where(ious <= thresh)[0]#判断阈值
        # print("idx", idx)
        index = index[idx + 1]  # because index start from 1 因为下标从1开始
    return keep #返回保存的值

def find_minandmax(x):
    area_min, area_max = x[0]*x[0+4], x[0]*x[0+4]
    minnum, maxnum = 0, 0
    for i in range(0, 4):
        if area_min > x[i]*x[i+4]:
            area_min = x[i]*x[i+4]
            minnum = i
    for i in range(0, 4):
        if area_max < x[i]*x[i+4]:
            area_max = x[i]*x[i+4]
            maxnum = i
    return minnum, maxnum


# 第2种：直接对检测框作五五开平均
def coop2(box1, box2):
    box1.x = (box1.x + box2.x) / 2
    box1.y = (box1.y + box2.y) / 2
    box1.z = (box1.z + box2.z) / 2
    box1.l = (box1.l + box2.l) / 2
    box1.w = (box1.w + box2.w) / 2
    box1.h = (box1.h + box2.h) / 2
    box1.yaw = (box1.yaw + box2.yaw) / 2
    for i in range(len(box1.bev_coord)):
        box1.bev_coord[i] = (box1.bev_coord[i] + box2.bev_coord[i]) / 2
    box1.score = (box1.score+box2.score)/2
    return box1

# 第3种：根据置信度，对IOU够高的数据作加权平均
def coop3(box1, box2):
    # 需要融合的参数有：中心点坐标xyz、尺寸lwh、yaw，然后需要重新测一遍iou
    box1.x = average_weight(box1.x, box2.x, box1.score, box2.score)
    box1.y = average_weight(box1.y, box2.y, box1.score, box2.score)
    box1.z = average_weight(box1.z, box2.z, box1.score, box2.score)
    box1.l = average_weight(box1.l, box2.l, box1.score, box2.score)
    box1.w = average_weight(box1.w, box2.w, box1.score, box2.score)
    box1.h = average_weight(box1.h, box2.h, box1.score, box2.score)
    box1.yaw = average_weight(box1.yaw, box2.yaw, box1.score, box1.score)
    for i in range(len(box1.bev_coord)):
        box1.bev_coord[i] = average_weight(box1.bev_coord[i], box2.bev_coord[i], box1.score, box2.score)
    box1.score = (box1.score+box2.score)/2
    return box1

# 第4种：直接取置信度最高的数据
def coop4(box1, box2):
    if box1.score >= box2.score:
        return box1
    else:
        return box2

# 置信度选取方法x2
# 第一种：mtp评分法
# 第二种：NDS评分法
def car_score(bboxes, car_score):
    for i in range(len(bboxes)):
        bboxes[i].score = bboxes[i].score * car_score
    return bboxes



def voc_ap(rec, prec):
    # 全点插值法
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))  #[0.  0.0666, 0.1333, 0.4   , 0.4666,  1.]
    mpre = np.concatenate(([0.], prec, [0.])) #[0.  1.,     0.6666, 0.4285, 0.3043,  0.]

    # compute the precision envelope
    # 计算出precision的各个断点(折线点)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  #[1.     1.     0.6666 0.4285 0.3043 0.    ]

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]  #precision前后两个值不一样的点
    # print(mrec[1:], mrec[:-1])
    # print(i) #[0, 1, 3, 4, 5]

    # AP= AP1 + AP2+ AP3+ AP4
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def gtrange_calculate(gt_bboxes):
    gtinrange_num = 0
    for i in range(len(gt_bboxes)):
        box = gt_bboxes[i]
        # ego_loc没有经过x10处理，而且z坐标为负数
        d = math.pow(pow(box.x-box.ego_loc[0]*10, 2)+pow(box.z+box.ego_loc[1]*10, 2), 0.5)
        if d <= 150:
            gtinrange_num +=1
    return gtinrange_num

def IOUmergedFP_calculate(bboxes, iou_score):
    invalid_bboxes = []
    for i in range(len(bboxes)):
        if bboxes[i].iou < iou_score:   # iou为0的pred_id也会被计数，要不要弄if bboxes[i].iou < iou_score and bboxes[i].iou != 0:
            invalid_bboxes.append(bboxes[i])
    FP_bboxes = []
    for i in range(len(invalid_bboxes)):
        FP_bboxes.append(invalid_bboxes[i].pred_id)
    # 计数FP_bboxes中有多少个不重复的数字
    # set()将列表中的元素无重复的抽取出来，赋值给另一个列表
    FP = len(set(FP_bboxes))
    return FP

def average_weight(x1, x2, weight1, weight2):
    return (x1 * weight1 + x2 * weight2) / (weight1 + weight2)      

def prepare_iou(bbox1, bbox2):
    bbox1_t, bbox2_t = [], []
    bbox1_t = image_vis.bev_corners(bbox1.x, bbox1.z, bbox1.l, bbox1.w, bbox1.yaw, 512, 768)
    bbox2_t = image_vis.bev_corners(bbox2.x, bbox2.z, bbox2.l, bbox2.w, bbox2.yaw, 512, 768)
    minnum1, maxnum1 = find_minandmax(bbox1_t)
    minnum2, maxnum2 = find_minandmax(bbox2_t)
    
    box1 = [bbox1_t[minnum1], bbox1_t[minnum1+4], bbox1_t[maxnum1], bbox1_t[maxnum1+4]]
    box2 = [bbox2_t[minnum2], bbox2_t[minnum2+4], bbox2_t[maxnum2], bbox2_t[maxnum2+4]]
    return iou(box1, box2)

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    # 如果两个检测框有交集则接着计算，否则直接判定iou=0
    if x_inter2 > x_inter1 and y_inter2 > y_inter1:
        width_inter = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
        area_inter = width_inter * height_inter

        width_box1 = abs(x2 - x1)
        height_box1 = abs(y2 - y1)
        width_box2 = abs(x4 - x3)
        height_box2 = abs(y4 - y3)
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        area_union = area_box1 + area_box2 - area_inter

        iou = area_inter / area_union
        # print(iou)
        return iou
    else:
        return 0

def find_minandmax(x):
    area_min, area_max = x[0]*x[0+4], x[0]*x[0+4]
    minnum, maxnum = 0, 0
    for i in range(0, 4):
        if area_min > x[i]*x[i+4]:
            area_min = x[i]*x[i+4]
            minnum = i
    for i in range(0, 4):
        if area_max < x[i]*x[i+4]:
            area_max = x[i]*x[i+4]
            maxnum = i
    return minnum, maxnum

def read_in_file(i, j, frame, mATE, mASE, mAOE, coiou_ap, NDS):
    if i == 0 and j == 0:
        with open("coop_IOU_evaluation1.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\t%s\n" % (frame, mATE, mASE, mAOE, coiou_ap, NDS, "score"+str(j+1)+"coop"+str(i+1)))
    elif i == 0 and j == 1:
        with open("coop_IOU_evaluation2.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\t%s\n" % (frame, mATE, mASE, mAOE, coiou_ap, NDS, "score"+str(j+1)+"coop"+str(i+1)))
    elif i == 1 and j == 0:
        with open("coop_IOU_evaluation3.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\t%s\n" % (frame, mATE, mASE, mAOE, coiou_ap, NDS, "score"+str(j+1)+"coop"+str(i+1)))
    elif i == 1 and j == 1:
        with open("coop_IOU_evaluation4.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\t%s\n" % (frame, mATE, mASE, mAOE, coiou_ap, NDS, "score"+str(j+1)+"coop"+str(i+1)))
    elif i == 2 and j == 0:
        with open("coop_IOU_evaluation5.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\t%s\n" % (frame, mATE, mASE, mAOE, coiou_ap, NDS, "score"+str(j+1)+"coop"+str(i+1)))
    elif i == 2 and j == 1:
        with open("coop_IOU_evaluation6.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\n" % (frame, mATE, mASE, mAOE, coiou_ap, NDS))
    elif i == 3 and j == 0:
        with open("coop_IOU_evaluation7.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\t%s\n" % (frame, mATE, mASE, mAOE, coiou_ap, NDS, "score"+str(j+1)+"coop"+str(i+1)))
    elif i == 3 and j == 1:
        with open("coop_IOU_evaluation8.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\t%s\n" % (frame, mATE, mASE, mAOE, coiou_ap, NDS, "score"+str(j+1)+"coop"+str(i+1)))
