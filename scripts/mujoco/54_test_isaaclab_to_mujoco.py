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

test
test


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



mujoco 模型参数




'''


# ===========================================================================
#  加载模型 (含 ray sensor 的 a2_box_ray.xml)
#  传感器顺序: sensordata[0:12]   = 12 个关节扭矩传感器
#              sensordata[12:199] = 187 个 rangefinder (高度扫描)
# ===========================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../taixi_model/A2/urdf/a2_box_ray.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

model.opt.timestep = sim_dt

base_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

# 扭矩传感器名称
torque_sensor_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    for i in range(12)  # 前 12 个是扭矩
]
print(f"扭矩传感器 ({len(torque_sensor_names)} 个): {torque_sensor_names}")

# ===========================================================================
#  Ray 参数 — 与 Isaac Lab RayCasterCfg 完全一致 (通过 XML rangefinder 实现)
#
#   坐标系 (Isaac Lab / 机器人坐标系):
#     +forward  (+x)  = 前       -forward  (-x)  = 后
#     +left     (+y)  = 左       -left     (-y)  = 右
#
#   pattern_cfg = GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
#     → forward 轴 1.6 m × left 轴 1.0 m, 间距 0.1 m
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

print(f"Grid rays: {NUM_FORWARD}×{NUM_LEFT} = {NUM_RAYS} total")
print(f"Ray sites: 从 XML data.site_xpos 读取位置, mj_ray(geomgroup[4]=1) 只命中地面")


# ---------------------------------------------------------------------------
#  核心: 高度扫描
#
#  使用 mj_ray() (而非 XML rangefinder)，因为:
#    1. rangefinder sensor 无法按 geomgroup 过滤 → 射线会命中 robot 自身碰撞体
#    2. mj_ray() 可以用 geomgroup[4]=1 只命中地面
#
#  ⚠️ 射线起点必须使用 yaw-only 旋转 (匹配 Isaac Lab ray_alignment="yaw"):
#    - origin_z 始终 = base_z + RAY_Z_OFFSET (不受 roll/pitch 影响)
#    - data.site_xpos 包含 roll/pitch → 不能直接用！
#    - 从 model.site_pos 读取 local offset, 手动做 yaw 旋转
# ---------------------------------------------------------------------------

# 解析 XML 中 ray_site_* 的 site ID 和 local offset
_ray_site_ids = []
_ray_local_offsets = []  # (fw_off, lt_off) in base_link frame
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    if name and name.startswith("ray_site_"):
        idx = int(name.split("_")[-1])
        local_pos = model.site_pos[i]  # (x, y, z) in parent (base_link) frame
        _ray_site_ids.append((idx, i, local_pos[0], local_pos[1]))
_ray_site_ids.sort(key=lambda x: x[0])  # 按编号排序: 0, 1, ..., 186
_ray_local_offsets = [(fw, lt) for _, _, fw, lt in _ray_site_ids]
_ray_site_ids = [sid for _, sid, _, _ in _ray_site_ids]
assert len(_ray_site_ids) == NUM_RAYS, f"Expected {NUM_RAYS} ray sites, found {len(_ray_site_ids)}"

print(f"Resolved {len(_ray_site_ids)} ray sites from XML")

# 预分配
_geomid_buf = np.array([-1], dtype=np.int32)
_ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # 世界系垂直向下
_geomgroup = np.zeros(6, dtype=np.uint8)
_geomgroup[4] = 1  # 只命中 group 4 (地面)


def cast_ray_grid(data):
    """对每个 ray site 发射 mj_ray，只命中地面 (group 4).

    射线起点使用 yaw-only 旋转 (匹配 Isaac Lab ray_alignment="yaw"):
      origin_world = base_pos + yaw_rot @ (fw_off, lt_off, 0) + (0, 0, RAY_Z_OFFSET)
    射线方向始终垂直向下 [0, 0, -1].

    Returns
    -------
    ray_distances : np.ndarray  shape (NUM_RAYS,)
        沿射线方向的距离。未命中 = -1.0。
    """
    base_pos = data.xpos[base_link_id]
    xmat = data.xmat[base_link_id]
    yaw = math.atan2(xmat[3], xmat[0])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    ray_distances = np.full(NUM_RAYS, -1.0, dtype=np.float64)

    for idx, (fw_off, lt_off) in enumerate(_ray_local_offsets):
        # yaw-only 旋转 (match Isaac Lab quat_apply_yaw)
        world_dx = fw_off * cos_yaw - lt_off * sin_yaw
        world_dy = fw_off * sin_yaw + lt_off * cos_yaw

        origin = np.array([
            base_pos[0] + world_dx,
            base_pos[1] + world_dy,
            base_pos[2] + RAY_Z_OFFSET,  # ← 始终 base_z + 20.0 (不受 roll/pitch 影响)
        ], dtype=np.float64)

        _geomid_buf[0] = -1
        dist = mujoco.mj_ray(model, data, origin, _ray_dir,
                             _geomgroup, 1, -1, _geomid_buf)
        if dist >= 0.0:
            ray_distances[idx] = dist

    return ray_distances


def get_height_scan_obs(data):
    """读取 ray site 位置 + mj_ray → Isaac Lab height_scanner 格式."""
    ray_distances = cast_ray_grid(data)

    # 处理未命中: 将 -1 替换为 RAY_Z_OFFSET
    ray_distances[ray_distances < 0] = RAY_Z_OFFSET

    # 转换为 Isaac Lab 格式
    height_scanner_obs = ray_distances - RAY_Z_OFFSET - 0.5
    height_scanner_obs = np.clip(height_scanner_obs, -1.0, 1.0)

    return height_scanner_obs


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


def compute_hit_points(data, ray_distances):
    """根据射线距离计算世界坐标命中点 (用于可视化).

    匹配 Isaac Lab ray_alignment="yaw":
      hit_point = yaw_only_origin + [0, 0, -1] * distance
    """
    base_pos = data.xpos[base_link_id]
    xmat = data.xmat[base_link_id]
    yaw = math.atan2(xmat[3], xmat[0])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    n = NUM_RAYS
    points    = np.full((n, 3), np.nan, dtype=np.float64)
    heights_z = np.full(n, np.nan, dtype=np.float64)

    for idx, (fw_off, lt_off) in enumerate(_ray_local_offsets):
        d = ray_distances[idx]
        if d < 0:
            continue

        # yaw-only origin_z
        origin_z = base_pos[2] + RAY_Z_OFFSET

        hit_world_z = origin_z - d
        world_dx = fw_off * cos_yaw - lt_off * sin_yaw
        world_dy = fw_off * sin_yaw + lt_off * cos_yaw
        points[idx] = [base_pos[0] + world_dx, base_pos[1] + world_dy, hit_world_z]
        heights_z[idx] = hit_world_z

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
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False  # 关闭黄色射线

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


                    # --- 高度扫描: 从 XML site 读取位置 + mj_ray 过滤 ground ---
                    ray_distances = cast_ray_grid(data)
                    height_scanner_obs = get_height_scan_obs(data)

                    # 可视化 (使用传感器原始距离)
                    hit_points, heights_z = compute_hit_points(data, ray_distances)
                    draw_height_dots(viewer, hit_points, heights_z)



                    if run_first_flag:

                        if step_count >= 199:
                            run_first_flag = False

                        # obs_compute
                        compute_obs(data,velocity_commands, action, height_scanner_obs, step_count * sim_dt, add_noise=False)

                        data.ctrl = init_joint_pos_isaaclab[isaaclab_to_mj_act]

                    else:


                        if step_count % decimation == 0:

                            step_flag = False
                            print(step_count)
                            # obs_compute
                            compute_obs(data,velocity_commands, action, height_scanner_obs, step_count * sim_dt, add_noise=False)
                            obs_tensor = torch.from_numpy(obs_fifo.get_fifo()).float().unsqueeze(0)

                            with torch.no_grad():
                                action_isaaclab = policy(obs_tensor).squeeze(0).numpy()

                            action = np.clip(action_isaaclab, -clip_actions, clip_actions)
                            target_q = init_joint_pos_isaaclab + action * action_scale


                            data.ctrl = target_q[isaaclab_to_mj_act]



                    mujoco.mj_step(model, data)

                    viewer.sync()


                    # 时间控制
                    elapsed = time.time() - step_start
                    if elapsed < sim_dt:
                        time.sleep(sim_dt - elapsed)
                    step_count += 1
    finally:
        monitor.stop()



if __name__ == "__main__":
    main()
