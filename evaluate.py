import math
from mmdet3d.core.visualizer import image_vis
import numpy as np
import cv2
import camera_script
import copy

def IOU_calculate(gt_boxes, pred_boxes):
    # 此时gt_boxes中的xyz为绝对坐标
    # pred_boxes中的xyz在image_vis的create_totalbev中就已经转换为绝对坐标了
    # 遍历每个预测框，对每个预测框计算其与所有真值框的iou
    for i in range(len(pred_boxes)):
        pred_boxes[i].iou = 0
        for j in range(len(gt_boxes)):
            # 取iou最大者的真值框id作为预测框的结果id
            iou = prepare_iou(pred_boxes[i], gt_boxes[j])
            if iou >= pred_boxes[i].iou:
                pred_boxes[i].iou = iou
                pred_boxes[i].pred_id = gt_boxes[j].num

def meIOU_calculate(gt_boxes, pred_boxes):
    for i in range(len(pred_boxes)):
        id = pred_boxes[i].pred_id - 1
        iou = prepare_iou(pred_boxes[i], gt_boxes[id])
        pred_boxes[i].iou = iou

# 未融合的冗余检测框的PR评估函数
def IOU_evaluate(pred_bboxes, iou_thres, frame, prec, rec, score_thres, score_method="score1", eval_method="eval1"):
    bboxes = copy.deepcopy(pred_bboxes)
    TP1, FP1, TN1, FN1 = 0, 0, 0, 0
    # 计算融合前的pr
    # 置信度大于score的为P，iou大于iou_thres的为TP，小于为FP；置信度小于score的为N，iou大于iou_thres的为FN，小于为TN
    # 又或是直接取检测框数目len(pred_bboxes)为GT，作为召回率的分母
    for i in range(len(bboxes)):
        if bboxes[i].score >= score_thres and bboxes[i].iou >= iou_thres:   TP1 += 1
        elif bboxes[i].score >= score_thres and bboxes[i].iou < iou_thres:  FP1 += 1
        elif bboxes[i].score < score_thres and bboxes[i].iou >= iou_thres:  FN1 += 1
        elif bboxes[i].score < score_thres and bboxes[i].iou < iou_thres:   TN1 += 1
    p1 = TP1 / (TP1 + FP1)
    p2 = p1
    if TP1 == 0 and FN1 == 0:
        r1 = 0
    else:
        r1 = TP1 / (TP1 + FN1)
    r2 = TP1 / len(bboxes)
    if eval_method == "eval1":
        precision, recall = p1, r1
    else:
        precision, recall = p2, r2
    # with open("prediction_IOU_evaluation.txt", "a") as file_read:
    #     file_read.write("%i\t%f\t%f\t%f\t%s\n" % (frame, score_thres, precision, recall, score_method+eval_method))
    prec.append(precision)
    rec.append(recall)

# 融合后的检测框的PR评估函数
def meIOU_evaluate(gt_bboxes, pred_bboxes, iou_thres, frame, prec, rec, score_thres, score_method="score1", merge_method="mergence1"):
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

# 融合前的检测框的mTP评估函数
def IOU_mTPeval(gt_boxes, pred_bboxes, frame):
    bboxes = copy.deepcopy(pred_bboxes)
    ATE, ASE, AOE = 0, 0, 0
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
        # with open("prediction_IOU_evaluation.txt", "a") as file_read:
        #     file_read.write("%i\t%i\t\t%f\t%f\t%f\t" % (frame, id+1, ate, ase, aoe))
    
    if len(bboxes) != 0:
        return ATE/len(bboxes), ASE/len(bboxes), AOE/len(bboxes)
    else:
        return 0, 0, 0

# 融合后的检测框的mTP评估函数
def meIOU_mTPeval(gt_boxes, pred_bboxes, frame, score_method="score1", merge_method="merge2"):
    bboxes = copy.deepcopy(pred_bboxes)
    ATE, ASE, AOE = 0, 0, 0
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
        # with open("merged_IOU_evaluation.txt", "a") as file_read:
        #     file_read.write("%i\t%i\t\t%f\t%f\t%f\t%s\n" % (frame, id+1, ate, ase, aoe, score_method+merge_method))
    
    if len(bboxes) != 0:
        return ATE/len(bboxes), ASE/len(bboxes), AOE/len(bboxes)
    else:
        return 0, 0, 0

# 包含融合操作与融合后指标计算操作这个函数原理上不应该改变预测框列表bboxes
def IOU_merge(pred_bboxes, iou_thres, merge_method="merge2"):
    bboxes = copy.deepcopy(pred_bboxes)
    # 首先消除检测框参数列表中，iou值为0的无效检测框
    valid_bboxes = []
    for i in range(len(bboxes)):
        if bboxes[i].iou >= iou_thres:
            valid_bboxes.append(bboxes[i])
    # 将pred_id相同的检测框进行融合
    final_boxes = []
    if len(valid_bboxes) == 1:
        final_boxes.append(valid_bboxes[0])
    else:
        for i in range(len(valid_bboxes)-1):
            if valid_bboxes[i].flag == "True":
                for j in range(i+1, len(valid_bboxes)):
                    if valid_bboxes[i].pred_id == valid_bboxes[j].pred_id:
                        # 挑选融合方案
                        if merge_method == "merge1":
                            valid_bboxes[i] = mergence1(valid_bboxes[i], valid_bboxes[j])
                        elif merge_method == "merge2":
                            valid_bboxes[i] = mergence2(valid_bboxes[i], valid_bboxes[j])
                        elif merge_method == "merge3":
                            valid_bboxes[i] = mergence3(valid_bboxes[i], valid_bboxes[j])
                        valid_bboxes[j].flag = "False"
                valid_bboxes[i].flag = "False"
                final_boxes.append(valid_bboxes[i])
    if len(valid_bboxes) != 0:
        if valid_bboxes[-1].flag == "True":
            final_boxes.append(valid_bboxes[-1])

    return final_boxes


def evaluate_script(gt_boxes, pred_boxes, iouap_list, frame):
    # 进行冗余检测框的评估
    IOUprec, IOUrec = [], []  # 准备计算所有iouscore下的精确度和召回率，并计算AP
    score_method = ["score1", "score2"]
    eval_method = ["eval1", "eval2"]
    # 隔离预测框参数原始列表和处理列表
    IOU_boxes = copy.deepcopy(pred_boxes)
    # 获取评判预测结果真假的指标数值
    IOU_calculate(gt_boxes, IOU_boxes) 
    score_list = []
    for i in range(len(score_method)):
        for j in range(len(eval_method)):
            # 置信度阈值应该是所有检测框score的从大到小递减，score从大到小排序，计算每个score下高于阈值或低于阈值的TP、FP
            if i == 0:  score = score1
            elif i == 1:score = score2
            IOU_boxes = score(IOU_boxes)
            score_list = []
            score_list = score_sort(score_list, IOU_boxes)
            for score in score_list:
                IOU_evaluate(IOU_boxes, 0.3, frame, IOUprec, IOUrec, score, score_method=score_method[i], eval_method=eval_method[j])
            # 计算AP
            iou_ap = voc_ap(IOUrec, IOUprec)
            iouap_list.append(iou_ap)
            with open("prediction_IOU_evaluation.txt", "a") as file_read:
                file_read.write("AP = %f\n" % (iou_ap))
    print("mAP1_IOU = %f, mAP2_IOU = %f, mAP3_IOU = %f, mAP4_IOU = %f\n" % 
            (AP_average(iouap_list)[0], AP_average(iouap_list)[1], AP_average(iouap_list)[2], AP_average(iouap_list)[3]))

# 方位评估法score2+评估方法eval1
def final_evaluate_scripts(gt_boxes, pred_boxes, iouap_list, frame, mATE_list, mASE_list, mAOE_list):
    # 进行冗余检测框的评估
    IOUprec, IOUrec = [], []  # 准备计算所有iouscore下的精确度和召回率，并计算AP
    # 隔离预测框参数原始列表和处理列表
    IOU_boxes = copy.deepcopy(pred_boxes)
    # 获取评判预测结果真假的指标数值
    IOU_calculate(gt_boxes, IOU_boxes) 
    score_list = []
    # 置信度阈值应该是所有检测框score的从大到小递减，score从大到小排序，计算每个score下高于阈值或低于阈值的TP、FP
    IOU_boxes = score2(IOU_boxes)
    mATE, mASE, mAOE = IOU_mTPeval(gt_boxes, IOU_boxes, frame)
    mATE_list.append(mATE)
    mASE_list.append(mASE)
    mAOE_list.append(mAOE)
    score_list = []
    score_list = score_sort(score_list, IOU_boxes)
    for score in score_list:
        IOU_evaluate(IOU_boxes, 0.3, frame, IOUprec, IOUrec, score, "score2", "eval1")
    # 计算AP
    iou_ap = voc_ap(IOUrec, IOUprec)
    iouap_list.append(iou_ap)
    NDS = NDS_calculate(iou_ap, mATE, mASE, mAOE)
    with open("prediction_IOU_evaluation.txt", "a") as file_read:
        file_read.write("%i\t%f\t%f\t%f\t%f\t%f\n" % (frame, mATE, mASE, mAOE, iou_ap, NDS))

def merge_script(gt_boxes, pred_boxes, mATE_list, mASE_list, mAOE_list, meiouap_list, frame):
    # 进行检测框的融合与评估
    meIOUprec, meIOUrec = [], []    # 准备计算所有iouscore下的精确度和召回率，并计算AP
    score_list = []
    score_method = ["score1", "score2"]
    merge_method = ["merge1", "merge2", "merge3"]
    eval_method = ["eval1", "eval2"]
    # 隔离预测框参数原始列表和处理列表
    IOU_boxes = copy.deepcopy(pred_boxes)
    # 获取评判预测结果真假的指标数值
    IOU_calculate(gt_boxes, IOU_boxes)
    # 2种置信度计算方法，两种融合方法，一共4种融合评估方案
    for i in range(len(score_method)):
        for j in range(len(merge_method)):
            # 获取融合后的检测框并进行iou重新计算，然后重新评估
            meIOU_boxes = IOU_merge(IOU_boxes, 0.3, merge_method=merge_method[j])
            meIOU_calculate(gt_boxes, meIOU_boxes)  # 融合后检测框的新iou
            # 置信度阈值应该是所有检测框score的从大到小递减，score从大到小排序，计算每个score下高于阈值或低于阈值的TP、FP
            if i == 0:
                meIOU_boxes = score1(meIOU_boxes)
                mATE, mASE, mAOE = meIOU_mTPeval(gt_boxes, meIOU_boxes, frame, score_method[i], merge_method[j])
                mergebev_printer(meIOU_boxes, frame, score_method[i], merge_method[j])
                mATE_list.append(mATE)
                mASE_list.append(mASE)
                mAOE_list.append(mAOE)
                # 重新计算mAP                
                score_list = score_sort(score_list, meIOU_boxes)
                for score in score_list:
                    meIOU_evaluate(gt_boxes, meIOU_boxes, 0.3, frame, meIOUprec, meIOUrec, score, score_method[i], merge_method[j])
                meiou_ap = voc_ap(meIOUrec, meIOUprec)
                meiouap_list.append(meiou_ap)
                if j == 0:
                    with open("prediction_IOU_evaluation.txt", "a") as file_read:
                        file_read.write("AP = %f\n" % (meiou_ap))
                NDS = (7*meiou_ap + (1-min(1,mATE)) + (1-min(1,mASE)) + (1-min(1,mAOE)))/10
                read_in_file(i, j, frame, mATE, mASE, mAOE, meiou_ap, NDS)
            elif i == 1:
                meIOU_boxes = score2(meIOU_boxes)
                mATE, mASE, mAOE = meIOU_mTPeval(gt_boxes, meIOU_boxes, frame, score_method[i], merge_method[j])
                mergebev_printer(meIOU_boxes, frame, score_method[i], merge_method[j])
                mATE_list.append(mATE)
                mASE_list.append(mASE)
                mAOE_list.append(mAOE)
                # 重新计算mAP
                score_list = score_sort(score_list, meIOU_boxes)
                for score in score_list:
                    meIOU_evaluate(gt_boxes, meIOU_boxes, 0.3, frame, meIOUprec, meIOUrec, score, score_method[i], merge_method[j])
                meiou_ap = voc_ap(meIOUrec, meIOUprec)
                meiouap_list.append(meiou_ap)
                # with open("prediction_IOU_evaluation.txt", "a") as file_read:
                #     file_read.write("AP = %f\n" % (meiou_ap))
                NDS = (7*meiou_ap + (1-min(1,mATE/10)) + (1-min(1,mASE)) + (1-min(1,mAOE)))/10
                read_in_file(i, j, frame, mATE, mASE, mAOE, meiou_ap, NDS)

# 方位评分法score2+绝对权重融合法merge3
def final_merge_script(gt_boxes, pred_boxes, mATE_list, mASE_list, mAOE_list, meiouap_list, frame, vehicle):
    # 进行检测框的融合与评估
    meIOUprec, meIOUrec = [], []    # 准备计算所有iouscore下的精确度和召回率，并计算AP
    score_list = []
    # 隔离预测框参数原始列表和处理列表
    IOU_boxes = copy.deepcopy(pred_boxes)
    # 获取评判预测结果真假的指标数值
    IOU_calculate(gt_boxes, IOU_boxes)
    # 获取融合后的检测框并进行iou重新计算，然后重新评估
    meIOU_boxes = IOU_merge(IOU_boxes, 0.3, merge_method="merge3")
    meIOU_calculate(gt_boxes, meIOU_boxes)  # 融合后检测框的新iou
    # 置信度阈值应该是所有检测框score的从大到小递减，score从大到小排序，计算每个score下高于阈值或低于阈值的TP、FP
    meIOU_boxes = score2(meIOU_boxes)
    mATE, mASE, mAOE = meIOU_mTPeval(gt_boxes, meIOU_boxes, frame, "score2", "merge3")
    if len(meIOU_boxes) != 0:
        mergebev_printer(meIOU_boxes, frame, vehicle)
    mATE_list.append(mATE)
    mASE_list.append(mASE)
    mAOE_list.append(mAOE)
    # 重新计算mAP
    score_list = score_sort(score_list, meIOU_boxes)
    for score in score_list:
        meIOU_evaluate(gt_boxes, meIOU_boxes, 0.3, frame, meIOUprec, meIOUrec, score, "score2", "merge3")
    meiou_ap = voc_ap(meIOUrec, meIOUprec)
    meiouap_list.append(meiou_ap)
    NDS = NDS_calculate(meiou_ap, mATE, mASE, mAOE)
    read_in_file(frame, mATE, mASE, mAOE, meiou_ap, NDS, vehicle)
    car_score1 = 1 - 0.4*(mATE/10) - 0.4*mASE - (mAOE/180)*0.2
    car_score2 = NDS
    return car_score1, car_score2, meIOU_boxes

def mergebev_printer(merge_bboxes, frame, vehicle):
    # 画图
    image = np.zeros((1024, 1024, 3), np.uint8)
    image.fill(255)
    font = cv2.FONT_HERSHEY_COMPLEX
    image_vis.create_egobev(image, merge_bboxes[0].ego_loc, "cam", font)
    for i in range(len(merge_bboxes)):
        u1, u2, u3, u4, v1, v2, v3, v4 = tuple(merge_bboxes[i].bev_coord)
        image_vis.bev_printer(image, int(u1), int(u2), int(u3), int(u4), int(v1), int(v2), int(v3), int(v4), font, merge_bboxes[i].num+1, merge_bboxes[i].score)
    camera_script.mkdir_folder('/home/piaozx/文档/carla-code/carlafisheye/', "merge_bev", vehicle)
    bev_filename = "/home/piaozx/文档/carla-code/carlafisheye/merge_bev/" + vehicle + "/frame" + str(frame) + "_BEV.png"
    cv2.imwrite(bev_filename, image)

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

# 融合方法x3
# 第1种：直接对检测框作五五开平均
def mergence1(box1, box2):
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

# 第2种：根据置信度，对IOU够高的数据作加权平均
def mergence2(box1, box2):
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

# 第3种：直接取置信度最高的数据
def mergence3(box1, box2):
    if box1.score >= box2.score:
        return box1
    else:
        return box2

# 置信度选取方法x2，在选出valid_bboxes之后操作
# 第一种：直接取原置信度
def score1(bboxes):
    return bboxes

# 第二种：按照距离和角度递减，中间/近距离最高，侧边/远处最低
def score2(bboxes):
    for i in range(len(bboxes)):
        # ego_loc没有经过x10处理，而且z坐标为负数，得出的d为实际距离的10倍
        d = math.pow(pow(bboxes[i].x-bboxes[i].ego_loc[0]*10, 2)+pow(bboxes[i].z+bboxes[i].ego_loc[1]*10, 2), 0.5)
        theta = abs(math.atan(bboxes[i].z/bboxes[i].x))  #theta为弧度值
        bboxes[i].score = (d/150)*0.5 + (theta*2/math.pi)*0.5
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

def find_max(ap_list):
    ap_max = 0
    for i in range(len(ap_list)):
        if ap_list[i] > ap_max:
            ap_max = ap_list[i]
    return ap_max

def read_in_file(frame, mATE, mASE, mAOE, meiou_ap, NDS, vehicle):
    # if i == 0 and j == 0:
    #     with open("merged_IOU_evaluation1.txt", "a") as file_read:
    #         file_read.write("%i\t%s\t\t%f\t%f\t%f\t%f\t%f\n" % (frame, "total", mATE, mASE, mAOE, meiou_ap, NDS))
    # elif i == 0 and j == 1:
    #     with open("merged_IOU_evaluation2.txt", "a") as file_read:
    #         file_read.write("%i\t%s\t\t%f\t%f\t%f\t%f\t%f\n" % (frame, "total", mATE, mASE, mAOE, meiou_ap, NDS))
    # elif i == 0 and j == 2:
    #     with open("merged_IOU_evaluation3.txt", "a") as file_read:
    #         file_read.write("%i\t%s\t\t%f\t%f\t%f\t%f\t%f\n" % (frame, "total", mATE, mASE, mAOE, meiou_ap, NDS))
    # elif i == 1 and j == 0:
    #     with open("merged_IOU_evaluation4.txt", "a") as file_read:
    #         file_read.write("%i\t%s\t\t%f\t%f\t%f\t%f\t%f\n" % (frame, "total", mATE, mASE, mAOE, meiou_ap, NDS))
    # elif i == 1 and j == 1:
    #     with open("merged_IOU_evaluation5.txt", "a") as file_read:
    #         file_read.write("%i\t%s\t\t%f\t%f\t%f\t%f\t%f\n" % (frame, "total", mATE, mASE, mAOE, meiou_ap, NDS))
    # elif i == 1 and j == 2:
    #     with open("merged_IOU_evaluation6.txt", "a") as file_read:
    #         file_read.write("%i\t%s\t\t%f\t%f\t%f\t%f\t%f\n" % (frame, "total", mATE, mASE, mAOE, meiou_ap, NDS))
    # with open("merged_IOU_evaluation.txt", "a") as file_read:
    #     file_read.write("%i\t%s\t\t%f\t%f\t%f\t%f\t%f\n" % (frame, "total", mATE, mASE, mAOE, meiou_ap, NDS))
    if vehicle == "vehicle1":
        with open("merged_IOU_evaluation1.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\n" % (frame, mATE, mASE, mAOE, meiou_ap, NDS))
    elif vehicle == "vehicle2":
        with open("merged_IOU_evaluation2.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\n" % (frame, mATE, mASE, mAOE, meiou_ap, NDS))
    elif vehicle == "vehicle3":
        with open("merged_IOU_evaluation3.txt", "a") as file_read:
            file_read.write("%i\t%f\t%f\t%f\t%f\t%f\n" % (frame, mATE, mASE, mAOE, meiou_ap, NDS))
