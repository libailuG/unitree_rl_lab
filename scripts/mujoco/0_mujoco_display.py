import mujoco
import mujoco.viewer
import numpy as np
import time
import math

import os

# 加载模型 (路径相对于脚本所在目录)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../taixi_model/A2/urdf/a2.xml")
# MODEL_PATH = '/home/libai/00_isaaclab/taixi_centaur01/model/taixi_centaur01/urdf_foot/taixi_centaur01.xml'

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)



# 设置视角
def set_camera(viewer):
    viewer.cam.azimuth = 90       # 水平旋转角度 (度)，0=从+x方向看
    viewer.cam.elevation = -20    # 俯仰角 (度)，负值=俯视，正值=仰视
    viewer.cam.distance = 5.0     # 相机距离目标点的距离
    viewer.cam.lookat[:] = [0.0, 0.0, 0.5]  # 相机注视点

# 仿真循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    set_camera(viewer)

    while viewer.is_running():


    # 持续下发位置指令 (PD 控制器会自动跟踪)
    # data.ctrl[:] = stand_pose

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)

viewer.close()
