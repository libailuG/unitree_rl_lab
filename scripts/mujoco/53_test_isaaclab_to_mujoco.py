import os
import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
import math

from libai_arraytfifo import ArrayFIFO
from libai_keyboard import TermiosKeyMonitor
torch.set_printoptions(precision=5, sci_mode=False)

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
    os.getcwd(), "logs/rsl_rl/", "taixi_a2_velocity_rough/2026-07-28_09-58-52/",
    "exported/", "policy_260728.pt"
)
print(f"policy_path: {policy_path}")
policy = torch.jit.load(policy_path)
policy.eval()







'''



mujoco 模型参数




'''


# 加载模型
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../taixi_model/A2/urdf/a2_box.xml")


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

model.opt.timestep = sim_dt

base_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

# ── 扭矩传感器名称 (按 MuJoCo sensordata 顺序) ──
torque_sensor_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    for i in range(model.nsensor)
]
print(f"扭矩传感器 ({model.nsensor} 个): {torque_sensor_names}")




# ===========================================================================
#  Ray-caster 参数 — 与 Isaac Lab RayCasterCfg 完全一致
#
#   坐标系 (Isaac Lab / 机器人坐标系):
#     +forward  (+x)  = 前       -forward  (-x)  = 后
#     +left     (+y)  = 左       -left     (-y)  = 右
#
#   pattern_cfg = GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
#     → forward 轴 1.6 m × left 轴 1.0 m, 间距 0.1 m
#
#   offset.pos  = (0, 0, 20)     → base_link 上方 20 m
#   ray_alignment = "full"        → 跟随 base_link 完整位姿 (roll/pitch/yaw),
#                                   射线方向始终为局部 -Z, 随底座一同旋转
#
#   网格展平: index = lt_idx * 17 + fw_idx  (left 外层, forward 内层)
#     index   0 = (-forward, -left) = 后右方 (最右下角)
#     index 186 = (+forward, +left) = 前左方 (最左上角)
# ===========================================================================

SIZE_FORWARD  = 1.6    # [m]  前/后 方向
SIZE_LEFT     = 1.0    # [m]  左/右 方向
RESOLUTION    = 0.1    # [m]  间距
RAY_Z_OFFSET  = 20.0   # [m]  base_link 上方偏移

NUM_FORWARD   = int(SIZE_FORWARD / RESOLUTION) + 1   # 17
NUM_LEFT      = int(SIZE_LEFT    / RESOLUTION) + 1   # 11
NUM_RAYS      = NUM_LEFT * NUM_FORWARD                # 187

# forward / left 轴上的偏移值
_FW_VALS = np.linspace(-SIZE_FORWARD / 2, SIZE_FORWARD / 2, NUM_FORWARD)
_LT_VALS = np.linspace(-SIZE_LEFT    / 2, SIZE_LEFT    / 2, NUM_LEFT)

# 展平顺序: left 外层, forward 内层 → index = lt_idx * NUM_FORWARD + fw_idx
# index   0 = (_FW_VALS[ 0], _LT_VALS[ 0]) = 后右方  (最右下角)
# index 186 = (_FW_VALS[16], _LT_VALS[10]) = 前左方  (最左上角)
GRID_OFFSETS = [(fw, lt) for lt in _LT_VALS for fw in _FW_VALS]

print(f"Grid rays: {NUM_FORWARD}×{NUM_LEFT} = {NUM_RAYS} total")



# ---------------------------------------------------------------------------
#  方位描述工具
# ---------------------------------------------------------------------------

def grid_pos_to_desc(lt_idx, fw_idx):
    """网格序号 → 机器人方位 (Isaac Lab: +forward=前, +left=左)."""
    # forward: 0=后, 8=中, 16=前
    fw_pos = fw_idx / (NUM_FORWARD - 1) * 2 - 1       # -1(后) ~ +1(前)
    if fw_pos < -0.3:
        fb = "后"
    elif fw_pos > 0.3:
        fb = "前"
    else:
        fb = ""
    # left: 0=右, 5=中, 10=左
    lt_pos = lt_idx / (NUM_LEFT - 1) * 2 - 1          # -1(右) ~ +1(左)
    if lt_pos < -0.3:
        lr = "右"
    elif lt_pos > 0.3:
        lr = "左"
    else:
        lr = ""

    return f"{fb}{lr}方"


# ---------------------------------------------------------------------------
#  核心: 高度扫描
# ---------------------------------------------------------------------------

def cast_ray_grid(model, data):
    """发射全部射线, 返回与 Isaac Lab 一致的 height_scanner 数据.

    射线方向与 base_link 的完整位姿 (roll/pitch/yaw) 保持一致,
    而非仅跟随 yaw。射线在 base_link 局部坐标系中始终沿 -Z 方向,
    在世界坐标系中随 base_link 旋转。

    Returns
    -------
    heights : np.ndarray  shape (NUM_LEFT, NUM_FORWARD)  (11, 17)
        ``heights[lt_idx, fw_idx]`` = 射线起点到命中点的距离.
        未命中 = -1.0.
    hit_z : np.ndarray  shape (NUM_LEFT, NUM_FORWARD)  (11, 17)
        命中点的世界 Z 坐标. 未命中 = NaN.
    """
    base_pos = data.xpos[base_link_id]               # (3,)  world position
    xmat     = data.xmat[base_link_id].copy()        # (9,)  row-major 3×3 rotation matrix
    xmat_3x3 = xmat.reshape(3, 3)                     # (3, 3)

    # 局部射线方向: 沿 base_link 的 -Z 轴, 经完整旋转后得到世界方向
    local_ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    ray_dir = xmat_3x3 @ local_ray_dir                 # (3,)  world ray direction

    geomid   = np.array([-1], dtype=np.int32)

    # 只命中 group 4（地面）
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[4] = 1

    heights = np.full(NUM_RAYS, -1.0, dtype=np.float64)
    hit_z   = np.full(NUM_RAYS, np.nan, dtype=np.float64)

    for idx, (fw_off, lt_off) in enumerate(GRID_OFFSETS):
        # 网格点在 base_link 局部坐标系中的偏移:
        #   fw_off → +X (forward),  lt_off → +Y (left),  RAY_Z_OFFSET → +Z (up)
        local_offset = np.array([fw_off, lt_off, RAY_Z_OFFSET], dtype=np.float64)
        world_offset = xmat_3x3 @ local_offset          # 用完整旋转矩阵变换到世界系

        origin = base_pos + world_offset
        geomid[0] = -1

        dist = mujoco.mj_ray(model, data, origin, ray_dir,
                             geomgroup, 1, -1, geomid)
        if dist >= 0.0:
            heights[idx] = dist
            # hit 点的世界 Z 坐标 = origin_z - dist * ray_dir_z (因为 dist 是沿 ray_dir 的距离)
            hit_z[idx]   = origin[2] - dist * ray_dir[2]

    return heights.reshape(NUM_LEFT, NUM_FORWARD), hit_z.reshape(NUM_LEFT, NUM_FORWARD)


def get_height_scan_obs(model, data):
    """Isaac Lab 标准 1D 观测: shape (NUM_RAYS,) = (187,)."""
    heights, _ = cast_ray_grid(model, data)
    flat = heights.ravel()
    flat[flat < 0] = RAY_Z_OFFSET
    return flat


# ---------------------------------------------------------------------------
#  可视化
# ---------------------------------------------------------------------------

def _height_to_rgb(z, z_min, z_max):
    """高度 → 颜色: 低→蓝, 中→绿, 高→红."""
    if z_max - z_min < 1e-6:
        return np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    t = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    if t < 0.5:
        r, g, b = 0.0, t * 2, 1.0 - t * 2
    else:
        r, g, b = (t - 0.5) * 2, 1.0 - (t - 0.5) * 2, 0.0
    return np.array([r, g, b, 1.0], dtype=np.float32)


def draw_height_dots(viewer, points, heights_z, radius=0.025):
    """按高度着色: 低→蓝, 中→绿, 高→红."""
    scene = viewer.user_scn
    scene.ngeom = 0
    dot_size = np.array([radius, radius, radius], dtype=np.float64)
    dot_mat  = np.eye(3, dtype=np.float64).reshape(-1)

    valid   = ~np.isnan(heights_z)
    z_valid = heights_z[valid]
    if len(z_valid) == 0:
        return
    z_min, z_max = z_valid.min(), z_valid.max()

    for i in range(min(len(points), scene.maxgeom)):
        if not valid[i]:
            continue
        color = _height_to_rgb(heights_z[i], z_min, z_max)
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            dot_size, points[i], dot_mat, color)
        scene.ngeom += 1


def heights_to_hit_points(model, data, heights_flat):
    """height distances → 世界坐标命中点 + 高度 Z (用于可视化).

    使用 base_link 的完整旋转矩阵 (roll/pitch/yaw), 与 cast_ray_grid 保持一致.
    """
    base_pos = data.xpos[base_link_id]
    xmat     = data.xmat[base_link_id].copy()
    xmat_3x3 = xmat.reshape(3, 3)

    # 世界射线方向 (与 cast_ray_grid 一致)
    local_ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    ray_dir = xmat_3x3 @ local_ray_dir

    n = len(GRID_OFFSETS)
    points     = np.full((n, 3), np.nan, dtype=np.float64)
    heights_z  = np.full(n, np.nan, dtype=np.float64)

    for idx, (fw_off, lt_off) in enumerate(GRID_OFFSETS):
        d = heights_flat[idx]
        if d < 0:
            continue
        local_offset = np.array([fw_off, lt_off, RAY_Z_OFFSET], dtype=np.float64)
        world_offset = xmat_3x3 @ local_offset
        origin = base_pos + world_offset
        # 命中点 = origin + d * ray_dir
        hit_point = origin + d * ray_dir
        points[idx]    = hit_point
        heights_z[idx] = hit_point[2]

    return points, heights_z


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



'''

obs_compute

'''

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
    base_lin_vel = data.qvel[0:3].copy()
    if add_noise:
        base_lin_vel += np.random.uniform(-0.1, 0.1, 3)
    
    # 2.0 角速度
    base_ang_vel = quat_rotate_inverse(base_quat, data.qvel[3:6])
    base_ang_vel = data.qvel[3:6].copy()
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
    # joint_pos_rel_scaled = data.qpos[0 + mj_qpos_to_isaaclab] - init_joint_pos_isaaclab

    # 6.0 joint_vel
    joint_vel_rel_scaled = data.qvel[6 + mj_qpos_to_isaaclab].copy()
    # joint_vel_rel_scaled = data.qvel[0 + mj_qpos_to_isaaclab]

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


# ===========================================================================
#  仿真循环
# ===========================================================================
def main():

    global action, velocity_commands, last_command_time


    step_count = 0
    run_first_flag = True
    monitor = TermiosKeyMonitor()

    monitor.start()

    step_flag = False

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer._opt.geomgroup[3] = 1   # 显示碰撞体 (group 3)
            viewer._opt.geomgroup[4] = 1   # 显示地面   (group 4)

            # 设置初始视角
            viewer.cam.lookat = [0.0, 0.0, 0.5]
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 90     # 从机器人左侧看
            viewer.cam.elevation = -15   # 略微俯视



            while viewer.is_running():

                monitor.frame()
                if monitor.was_just_pressed('s'):
                    step_flag = True
                if monitor.was_just_pressed('q'):
                    break


                if step_flag:
                    step_start = time.time()


                    # --- Isaac Lab 标准 height_scanner 观测 ---
                    heights_2d, hit_z_2d = cast_ray_grid(model, data)
                    # 1D 观测: shape (187,)
                    # height_scanner_obs = RAY_Z_OFFSET - heights_2d.ravel() - 0.5
                    # dist = base_z + RAY_Z_OFFSET - ground_z
                    # ∴ base_z - ground_z - 0.5 = dist - RAY_Z_OFFSET - 0.5
                    height_scanner_obs = heights_2d.ravel() - RAY_Z_OFFSET - 0.5
                    # 还需要 clip (Isaac Lab 有 clip=(-1.0, 1.0))
                    height_scanner_obs = np.clip(height_scanner_obs, -1.0, 1.0)

                    # 可视化
                    hit_points, heights_z = heights_to_hit_points(model, data, heights_2d.ravel())
                    draw_height_dots(viewer, hit_points, heights_z)



                    if run_first_flag:

                        if step_count >= 199:
                            run_first_flag = False

                        # obs_compute
                        compute_obs(data,velocity_commands, action, height_scanner_obs, step_count * sim_dt, add_noise=False)
                        # print(obs_fifo.get_fifo().shape)

                        data.ctrl = init_joint_pos_isaaclab[isaaclab_to_mj_act]

                    else:


                        if step_count % decimation == 0:
                            step_flag = False
                            print(step_count)
                            # obs_compute
                            compute_obs(data,velocity_commands, action, height_scanner_obs, step_count * sim_dt, add_noise=False)
                            # print(obs_fifo.get_fifo().shape)
                            obs_tensor = torch.from_numpy(obs_fifo.get_fifo()).float().unsqueeze(0)

                            height_scanner_obs_index_start = (3 + 3 + 3 + 3 + 12 + 12 + 12) * 3
                            height_scanner_obs_index_end = height_scanner_obs_index_start + 187 * 3
                            # obs_tensor[:, height_scanner_obs_index_start:height_scanner_obs_index_end] = 0.51503
            #                 obs_tensor[:, height_scanner_obs_index_start:height_scanner_obs_index_end] = torch.tensor([
            #                     0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51478,      0.51478,      0.51478,      0.51478,
            #   0.51478,      0.51489,      0.51489,      0.51490,      0.51490,
            #   0.51490,      0.51489,      0.51489,      0.51490,      0.51490,
            #   0.51490,      0.51489,      0.51489,      0.51489,      0.51490,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51490,      0.51490,      0.51490,      0.51489,      0.51489,
            #   0.51490,      0.51489,      0.51490,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51490,      0.51490,
            #   0.51490,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51490,      0.51489,      0.51489,      0.51489,      0.51490,
            #   0.51490,      0.51489,      0.51489,      0.51489,      0.51490,
            #   0.51490,      0.51490,      0.51489,      0.51489,      0.51490,
            #   0.51490,      0.51489,      0.51489,      0.51489,      0.51490,
            #   0.51490,      0.51489,      0.51490,      0.51489,      0.51489,
            #   0.51489,      0.51490,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51490,      0.51489,      0.51490,      0.51489,
            #   0.51489,      0.51490,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51490,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51490,      0.51490,      0.51489,
            #   0.51489,      0.51489,      0.51490,      0.51489,      0.51489,
            #   0.51490,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51490,      0.51490,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51490,      0.51489,      0.51489,
            #   0.51490,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51490,      0.51490,      0.51489,      0.51489,
            #   0.51489,      0.51490,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51489,      0.51489,
            #   0.51489,      0.51490,      0.51489,      0.51489,      0.51490,
            #   0.51490,      0.51489,      0.51489,      0.51490,      0.51489,
            #   0.51489,      0.51489,      0.51489,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503,
            #   0.51503,      0.51503,      0.51503,      0.51503,      0.51503
            #                 ])

                            base_lin_vel_index_start = 0
                            base_ang_vel_index_end = 3 * 3
                            # obs_tensor[:, base_lin_vel_index_start:base_ang_vel_index_end] = torch.tensor([0.01987,0.00008,0.01101,0.02104,0.00010,0.01112,0.02195,0.00000,0.00979])

                            base_ang_vel_index_start = 3 * 3
                            base_ang_vel_index_end = (3+3) * 3
                            # obs_tensor[:, base_ang_vel_index_start:base_ang_vel_index_end] = torch.tensor([0.00185,-0.04749,-0.00082,0.00140,-0.05941,-0.00140,0.00111,-0.06832,-0.00078])

                            projected_gravity_index_start = (3+3) * 3
                            projected_gravity_index_end = (3+3+3) * 3
                            # obs_tensor[:, projected_gravity_index_start:projected_gravity_index_end] = torch.tensor([0.04080,-0.00003,-0.99917,0.03979,-0.00005,-0.99921,0.03858,-0.00008,-0.99926])


                            joint_pos_rel_scaled_index_start = (3+3+3 + 3) * 3
                            joint_pos_rel_scaled_index_end = (3+3+3+3 + 12) * 3
#                             obs_tensor[:, joint_pos_rel_scaled_index_start:joint_pos_rel_scaled_index_end] = torch.tensor([-0.00001,      0.00014,     -0.00994,      0.00980,     -0.02293,     -0.02291,      0.00828,      0.00838,     -0.02618,     -0.02621,      0.00036,     -0.00058,
# -0.00008,      0.00012,     -0.00995,      0.00979,     -0.02073,     -0.02073,      0.00882,      0.00896,     -0.02790,     -0.02797,      0.00039,     -0.00058,
# -0.00014,      0.00011,     -0.00995,      0.00979,     -0.01816,     -0.01819,      0.00947,      0.00969,     -0.02990,     -0.03007,      0.00041,     -0.00065])


                            joint_vel_index_start = (3+3+3+3 + 12) * 3
                            joint_vel_index_end = (3+3+3+3+12 + 12) * 3
#                             obs_tensor[:, joint_vel_index_start:joint_vel_index_end] = torch.tensor([-0.00820,     -0.00375,      0.01384,     -0.00091,      0.21359,      0.20537,     -0.19342,     -0.16165,     0.05376,     -0.01376,      0.02650,      0.00152,
# -0.00440,      0.00196,      0.00432,     -0.00588,      0.24034,      0.22099,     -0.19694,     -0.14167,     0.04033,     -0.04490,      0.00856,     -0.02797,
# -0.00064,      0.00213,     -0.01536,     -0.02856,      0.26496,      0.24760,     -0.20780,     -0.15782,     0.02323,     -0.04226,     -0.00376,      0.00137     
# ])
                            gait_phase_index_start = (3+3+3+3+12+12 + 12 + 187) * 3
                            gait_phase_index_end = (3+3+3+3+12+12+12 + 187 + 3) * 3
#                             obs_tensor[:, gait_phase_index_start:gait_phase_index_end] = torch.tensor([0.95106,      0.30902,
# 0.98769,      0.15643,
# 1.00000,      0.00000])


                            with torch.no_grad():
                                action_isaaclab = policy(obs_tensor).squeeze(0).numpy()

                            action = np.clip(action_isaaclab, -clip_actions, clip_actions)
                            target_q = init_joint_pos_isaaclab + action * action_scale
                            print(obs_tensor)

                            print(target_q*180/np.pi)


                            data.ctrl = target_q[isaaclab_to_mj_act]



                    mujoco.mj_step(model, data)

                    viewer.sync()


                    # 时间控制
                    elapsed = time.time() - step_start
                    if elapsed < sim_dt:
                        time.sleep(sim_dt - elapsed)
                    step_count += 1
                    if step_count % 100 == 0:
                        # print(height_scanner_obs[0],height_scanner_obs[16],height_scanner_obs[170],height_scanner_obs[186])
                        pass
    finally:
        monitor.stop()



if __name__ == "__main__":
    main()


# with mujoco.viewer.launch_passive(model, data) as viewer:
#     viewer._opt.geomgroup[3] = 1   # 显示碰撞体 (group 3)
#     viewer._opt.geomgroup[4] = 1   # 显示地面   (group 4)

#     step = 0
#     while viewer.is_running():
#         mujoco.mj_step(model, data)

#         # --- Isaac Lab 标准 height_scanner 观测 ---
#         heights_2d, hit_z_2d = cast_ray_grid(model, data)

#         # 1D 观测: shape (187,)
#         obs = RAY_Z_OFFSET - heights_2d.ravel() - 0.5
#         # -------------------------------------------

#         # 可视化
#         hit_points, heights_z = heights_to_hit_points(model, data, heights_2d.ravel())
#         draw_height_dots(viewer, hit_points, heights_z)

#         viewer.sync()
#         time.sleep(0.001)

#         # 每 100 步打印
#         step += 1
#         if step % 100 == 0:
#             flat_min = np.argmin(obs)
#             flat_max = np.argmax(obs)
#             lt_min, fw_min = np.unravel_index(flat_min, heights_2d.shape)
#             lt_max, fw_max = np.unravel_index(flat_max, heights_2d.shape)

#             print(f"\n--- step {step} ---")
#             print(f"  obs.max = {obs[flat_max]:.3f}  @ idx={flat_max}  [{grid_pos_to_desc(lt_max, fw_max)}]"
#                   f"  (lt={lt_max}, fw={fw_max})")
#             print(f"------------------------------\n")
