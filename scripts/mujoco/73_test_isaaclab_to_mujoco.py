"""
view_launch_ray_caster.py
=========================
基于 MuJoCo 的 ray_caster 传感器演示程序。

功能：
  - 加载插件、模型，启动仿真窗口
  - 终端每 100 步打印一次命中点的 z 坐标范围

运行：
  conda activate env_isaaclab_0
  cd demo/Python
  python3 view_launch_ray_caster.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import math
import os
import torch
import time

from libai_arraytfifo import ArrayFIFO
from libai_keyboard import TermiosKeyMonitor
torch.set_printoptions(precision=5, sci_mode=False)

# ============================================================
# 加载插件
# ============================================================
# 从 conda 环境中的 mujoco/plugin/ 目录加载插件（通用路径，不依赖项目路径）
import os as _os
_mujoco_dir = _os.path.dirname(mujoco.__file__)
_plugin_path = _os.path.join(_mujoco_dir, 'plugin', 'libsensor_raycaster.so')
try:
    mujoco.mj_loadPluginLibrary(_plugin_path)
except mujoco.FatalError as e:
    if 'already registered' not in str(e):
        raise

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../taixi_model/A2/urdf/a2_box_plugin.xml")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

def get_ray_caster_info(m, d, sensor_name):
    data_ps = []
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        return 0, 0, data_ps
    plugin_id = m.sensor_plugin[sensor_id]
    state_idx = m.plugin_stateadr[plugin_id]
    state_num = m.plugin_statenum[plugin_id]
    for i in range(state_idx + 2, state_idx + state_num, 2):
        if i + 1 < len(d.plugin_state):
            data_ps.append((int(d.plugin_state[i]), int(d.plugin_state[i + 1])))
    h = int(d.plugin_state[state_idx]) if state_idx < len(d.plugin_state) else 0
    v = int(d.plugin_state[state_idx + 1]) if state_idx + 1 < len(d.plugin_state) else 0
    return h, v, data_ps


sensor_name = "raycaster"
h_rays, v_rays, pairs = get_ray_caster_info(model, data, sensor_name)
print(f"=== Sensor: {sensor_name} ===")
print(f"h_rays={h_rays}, v_rays={v_rays}")
for i, (offset, size) in enumerate(pairs):
    print(f"  block[{i}]: offset={offset}, size={size}")
print("仿真运行中，每 100 步输出一次数据...\n")


def get_ray_caster_obs(data, sensor_name):
    sensor_data = data.sensor(sensor_name).data.copy()
    # ---------------------------------------------------------------
    # 数据块布局 (由 sensor_data_types 决定):
    #   block 0: inv_image_noise   (h*w floats)
    #   block 1: image_noise       (h*w floats)
    #   block 2: pos_w             (h*w*3 floats, 世界坐标)
    #   block 3: pos_b             (h*w*3 floats, 传感器本体坐标)
    # ---------------------------------------------------------------

    # ---------- 相对坐标 (pos_b)：障碍物相对于传感器的位置 ----------
    # 展平数组, 长度 = h * v * 3, 每 3 个连续值为 (x, y, z)
    #    x: 前方 (forward)
    #    y: 左方 (left)
    #    z: 下方 (down, 传感器的 z 轴朝下)
    # 未命中 = NaN
    offset_b, size_b = pairs[3]
    pos_b_flat = np.array(sensor_data[offset_b:offset_b + size_b], dtype=np.float32)

    # ------ 重映射到 (y_index, x_index) 网格 ------
    # pos_b 展平布局 (行优先): index = row * h_rays + col
    #   row=0  →  y=+0.5 (前)        col=0  →  x=-0.8 (左)
    #   row=10 →  y=-0.5 (后)        col=16 →  x=+0.8 (右)
    #
    # 展平 → reshape(h_rays*v_rays, 3) → reshape(v_rays, h_rays, 3)
    pos_b_3d = pos_b_flat.reshape(v_rays, h_rays, 3)   # [row, col, xyz]

    # z 通道: pos_b_3d[row, col, 2] → 障碍物在传感器下的高度
    height_map = pos_b_3d[:, :, 2].copy()              # [row, col]

    # ------ 翻转 y 轴: row0=后(y=-0.5), row10=前(y=+0.5) ------
    # 上面的 height_map[0,:] 对应 y=+0.5 (前), height_map[10,:] 对应 y=-0.5 (后)
    # flipud 后: height_map[0,:] → y=-0.5 (后), height_map[10,:] → y=+0.5 (前)
    height_map = np.flipud(height_map)

    height_map = -height_map - 0.5

    valid_z = height_map[np.isfinite(height_map)]  # index = y_idx * 17 + x_idx
    return valid_z            







'''

## isaaclab


# 关节顺序
#   [  0] left_hip_roll_joint
#   [  1] right_hip_roll_joint
#   [  2] left_hip_yaw_joint
#   [  3] right_hip_yaw_joint
#   [  4] left_hip_pitch_joint
#   [  5] right_hip_pitch_joint
#   [  6] left_knee_joint
#   [  7] right_knee_joint
#   [  8] left_ankle_pitch_joint
#   [  9] right_ankle_pitch_joint
#   [ 10] left_ankle_roll_joint
#   [ 11] right_ankle_roll_joint


'''


isaaclab_joint_order = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]



init_joint_pos_isaaclab = np.array([
     0.0,
     0.0,
     0.0,
     0.0,
     -10.0 / 180.0 * math.pi,
     -10.0 / 180.0 * math.pi,
     20.0 / 180.0 * math.pi,
     20.0 / 180.0 * math.pi,
     -10.0 / 180.0 * math.pi,
     -10.0 / 180.0 * math.pi,
     0.0,
     0.0,
])


joint_num = len(isaaclab_joint_order)


# 动作系数 动作范围
action_scale = 0.25
clip_actions = 12.56

# 模拟时间步长 内部时间步长
sim_dt = 0.005
decimation = 4

# obs
gait_period = 0.8

obs_num_arrays = 9
obs_arrar_size = 3 + 3 + 3 + 3 + 12 + 12 + 12 + 187 + 2
obs_arrar_size = 237
obs_history_length = 3
obs_fifo = ArrayFIFO(num_groups=obs_history_length, num_arrays=obs_num_arrays)

obs = np.zeros((obs_history_length * obs_arrar_size))

velocity_commands = np.array([0.0, 0.0, 0.0])


action = np.zeros(joint_num)

policy_path = os.path.join(
    os.getcwd(), "logs/rsl_rl/", "taixi_a2_velocity_rough/2026-07-29_13-49-49/",
    "exported/", "policy_260729.pt"
)
print(f"policy_path: {policy_path}")
policy = torch.jit.load(policy_path)
policy.eval()



'''

isaaclab to mujoco

'''

mj_qpos_order = []
for i in range(model.njnt):
    jnt_type = model.jnt_type[i]
    if jnt_type != mujoco.mjtJoint.mjJNT_FREE:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        mj_qpos_order.append(name)

print(f"MuJoCo qpos 顺序: {mj_qpos_order}")

# MuJoCo actuator 顺序
mj_act_order = []
for i in range(model.nu):
    jnt_id = model.actuator_trnid[i][0]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    mj_act_order.append(jnt_name)

print(f"MuJoCo actuator 顺序: {mj_act_order}")

# ============================================================
# 建立 remap 映射
# ============================================================
# 将 MuJoCo qpos/qvel 数据重排为 IsaacLab 顺序 (供策略输入)
mj_qpos_to_isaaclab  = np.array([mj_qpos_order.index(n) for n in isaaclab_joint_order], dtype=int)
# 将 MuJoCo actuator 数据重排为 IsaacLab 顺序 (供 last_action)
mj_act_to_isaaclab  = np.array([mj_act_order.index(n) for n in isaaclab_joint_order], dtype=int)
# 将 IsaacLab 顺序 (策略输出 action) 重排为 MuJoCo actuator 顺序 (供 data.ctrl)
isaaclab_to_mj_act  = np.array([isaaclab_joint_order.index(n) for n in mj_act_order], dtype=int)

print(f"MuJoCo qpos 顺序映射到 IsaacLab 顺序为: {mj_qpos_to_isaaclab}")
print(f"MuJoCo actuator 顺序映射到 IsaacLab 顺序为: {mj_act_to_isaaclab}")
print(f"IsaacLab actuator 顺序映射到 MuJoCo 顺序为: {isaaclab_to_mj_act}")



def quat_rotate_inverse(q, v):
    """用四元数 q (w, x, y, z) 的逆旋转向量 v (世界→机体)."""
    q_w, q_x, q_y, q_z = q
    qc_w, qc_x, qc_y, qc_z = q_w, -q_x, -q_y, -q_z
    # tmp = qc * v
    tmp_w = -qc_x * v[0] - qc_y * v[1] - qc_z * v[2]
    tmp_x =  qc_w * v[0] + qc_y * v[2] - qc_z * v[1]
    tmp_y =  qc_w * v[1] + qc_z * v[0] - qc_x * v[2]
    tmp_z =  qc_w * v[2] + qc_x * v[1] - qc_y * v[0]
    # result = tmp * q
    return np.array([
        tmp_w * q_x + tmp_x * q_w + tmp_y * q_z - tmp_z * q_y,
        tmp_w * q_y + tmp_y * q_w + tmp_z * q_x - tmp_x * q_z,
        tmp_w * q_z + tmp_z * q_w + tmp_x * q_y - tmp_y * q_x,
    ])


def compute_obs(data,velocity_commands,last_action, height_scanner_obs, global_time, add_noise=False):


    # 1.0 线速度
    base_quat = data.qpos[3:7].copy()
    base_lin_vel = quat_rotate_inverse(base_quat, data.qvel[0:3])
    if add_noise:
        base_lin_vel += np.random.uniform(-0.1, 0.1, 3)

    # 2.0 角速度
    base_ang_vel = quat_rotate_inverse(base_quat, data.qvel[3:6])
    if add_noise:
        base_ang_vel += np.random.uniform(-0.2, 0.2, 3)

    # 3.0 重力投影
    base_quat = data.qpos[3:7].copy()
    projected_gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
    if add_noise:
        projected_gravity += np.random.uniform(-0.05, 0.05, 3)

    # 4.0 velocity_commands

    # 5.0 joint_pos
    joint_pos_rel_scaled = data.qpos[7 + mj_qpos_to_isaaclab] - init_joint_pos_isaaclab

    # 6.0 joint_vel
    joint_vel_rel_scaled = data.qvel[6 + mj_qpos_to_isaaclab].copy()

    # 7.0 last_action

    # 8.0 height_scanner

    # 9.0 gait_phase
    gait_phase = np.zeros(2, dtype=np.float32)
    gait_phase[0] = np.sin(np.pi * 2.0 * ((global_time % gait_period) / gait_period))
    gait_phase[1] = np.cos(np.pi * 2.0 * ((global_time % gait_period) / gait_period))



    # over
    obs_fifo.push(base_lin_vel,
                  base_ang_vel,
                  projected_gravity,
                  velocity_commands,
                  joint_pos_rel_scaled,
                  joint_vel_rel_scaled,
                  last_action,
                  height_scanner_obs,
                  gait_phase)

    pass





step_count = 0
run_first_flag = True

 

def step_callback(_model, _data):
    global step_count, sensor_name, sensor_link_name
    step_count += 1

    if run_first_flag:
        _data.ctrl = init_joint_pos_isaaclab[isaaclab_to_mj_act]





# ============================================================
# 6. 注册回调 + 启动仿真窗口（内部是无限循环，直到关闭窗口）
# ============================================================
mujoco.set_mjcb_passive(step_callback)
mujoco.viewer.launch(model, data)
# mujoco.viewer.launch(model, data)._opt.geomgroup[3] = 1
# mujoco.viewer.launch_passive(model, data) as viewer
