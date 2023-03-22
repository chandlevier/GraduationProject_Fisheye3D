# GraduationProject_Fisheye3D
主要应用算法包括：

1. 图像获取与数据集搭建

		a. 鱼眼相机建模算法cubemap
		b. 场景搭建与图像获取npc.py, camera_script.py
		c. 图像处理与数据集搭建cube2fisheye.py, json.py, stitch.py

2. 基于鱼眼图像的三维目标检测算法

		a.mmdetection3d
		b.合并四个鱼眼相机鸟瞰图结果的算法
		c.预测结果与真值比较算法mix.py
3. 基于后期协同的感知结果融合算法

		a.利用不同车辆的坐标转换关系，获取多个车辆的预测3Dbbox参数，最后整合到一张图像上merge.py
