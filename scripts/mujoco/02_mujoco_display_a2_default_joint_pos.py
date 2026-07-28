import mujoco
import mujoco.viewer
import numpy as np
import time
import math

import os

# 加载模型 (路径相对于脚本所在目录)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../taixi_model/A2/urdf/a2_default_joint_pos.xml")
# MODEL_PATH = '/home/libai/00_isaaclab/taixi_centaur01/model/taixi_centaur01/urdf_foot/taixi_centaur01.xml'

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
base_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")


# 设置视角
def set_camera(viewer):
    viewer.cam.azimuth = 90       # 水平旋转角度 (度)，0=从+x方向看
    viewer.cam.elevation = -20    # 俯仰角 (度)，负值=俯视，正值=仰视
    viewer.cam.distance = 5.0     # 相机距离目标点的距离
    viewer.cam.lookat[:] = [0.0, 0.0, 0.5]  # 相机注视点

# 仿真循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    set_camera(viewer)

    viewer._opt.geomgroup[3] = 1   # 显示碰撞体 (group 3)
    viewer._opt.geomgroup[4] = 1   # 显示地面   (group 4)


    while viewer.is_running():


    # 持续下发位置指令 (PD 控制器会自动跟踪)
    # data.ctrl[:] = stand_pose
        robot_pos = data.xpos[base_link_id]
        print(f" robot world pos = ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})")
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)

viewer.close()
