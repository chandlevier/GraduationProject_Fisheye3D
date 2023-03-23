# GraduationProject_Monodet3D
存放基于mmdetection3d运行的目标检测算法

monodet3d.py为主程序，遍历鱼眼图像文件夹及标注文件夹，生成对应的检测结果文件夹

其中对image_vis.py加以改进，增加了获取边界框参数(xyz, hlw, yaw)并生成对应鸟瞰图的功能

inference(备份，可输出bev).py、show_result_bev.py、image_vis_bev.py中将一个鱼眼相机的鸟瞰图生成与存储功能放在了inference.py文件中(因为一开始在image_vis中生成鸟瞰图总报错，后来发现是调用函数时函数输入参数没写对)

inference_1fish.py、show_result_1fish.py、image_vis_1fish.py中将一个鱼眼相机的鸟瞰图生成与存储功能放在了image_vis中

inference.py、show_result.py、image_vis.py则是在image_vis中生成并存储四个鱼眼相机的鸟瞰图
