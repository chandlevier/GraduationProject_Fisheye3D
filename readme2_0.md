每辆车上有4个鱼眼相机fisheyex4.py
每个鱼眼相机由5个针孔相机组成，函数打包于camera_script.py中

针孔图像转换为鱼眼图像：tofisheyex4_front/right/left/back.py (调用cube2fisheye.py)
四个视角的鱼眼图像拼接在一起：stitch.py

