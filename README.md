# GraduationProject_Monodet3D
存放基于mmdetection3d运行的目标检测算法

monodet3d.py为主程序，遍历鱼眼图像文件夹及标注文件夹，生成对应的检测结果文件夹

其中对image_vis.py加以改进，增加了获取边界框参数(xyz, hlw, yaw)并生成对应鸟瞰图的功能
