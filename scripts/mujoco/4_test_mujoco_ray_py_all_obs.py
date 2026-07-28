import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import os

# 加载模型
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../taixi_model/A2/urdf/a2_box.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

base_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

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
#   ray_alignment = "yaw"         → 跟随 yaw, 射线垂直向下
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

    Returns
    -------
    heights : np.ndarray  shape (NUM_LEFT, NUM_FORWARD)  (11, 17)
        ``heights[lt_idx, fw_idx]`` = 射线起点到命中点的距离.
        未命中 = -1.0.
    hit_z : np.ndarray  shape (NUM_LEFT, NUM_FORWARD)  (11, 17)
        命中点的世界 Z 坐标. 未命中 = NaN.
    """
    base_pos = data.xpos[base_link_id]
    ray_dir  = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    geomid   = np.array([-1], dtype=np.int32)

    # 提取 base_link yaw
    xmat    = data.xmat[base_link_id]
    yaw     = math.atan2(xmat[1], xmat[0])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    # 只命中 group 4（地面）
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[4] = 1

    heights = np.full(NUM_RAYS, -1.0, dtype=np.float64)
    hit_z   = np.full(NUM_RAYS, np.nan, dtype=np.float64)

    for idx, (fw_off, lt_off) in enumerate(GRID_OFFSETS):
        # 用 yaw 旋转网格偏移量
        world_dx = fw_off * cos_yaw - lt_off * sin_yaw
        world_dy = fw_off * sin_yaw + lt_off * cos_yaw

        origin = np.array([
            base_pos[0] + world_dx,
            base_pos[1] + world_dy,
            base_pos[2] + RAY_Z_OFFSET,
        ], dtype=np.float64)
        geomid[0] = -1

        dist = mujoco.mj_ray(model, data, origin, ray_dir,
                             geomgroup, 1, -1, geomid)
        if dist >= 0.0:
            heights[idx] = dist
            hit_z[idx]   = origin[2] - dist

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
    """height distances → 世界坐标命中点 + 高度 Z (用于可视化)."""
    base_pos = data.xpos[base_link_id]
    xmat     = data.xmat[base_link_id]
    yaw      = math.atan2(xmat[1], xmat[0])
    cos_yaw  = math.cos(yaw)
    sin_yaw  = math.sin(yaw)

    n = len(GRID_OFFSETS)
    points     = np.full((n, 3), np.nan, dtype=np.float64)
    heights_z  = np.full(n, np.nan, dtype=np.float64)
    origin_z   = base_pos[2] + RAY_Z_OFFSET

    for idx, (fw_off, lt_off) in enumerate(GRID_OFFSETS):
        d = heights_flat[idx]
        if d < 0:
            continue
        world_dx = fw_off * cos_yaw - lt_off * sin_yaw
        world_dy = fw_off * sin_yaw + lt_off * cos_yaw
        z = origin_z - d
        points[idx]    = [base_pos[0] + world_dx, base_pos[1] + world_dy, z]
        heights_z[idx] = z

    return points, heights_z


# ===========================================================================
#  仿真循环
# ===========================================================================

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer._opt.geomgroup[3] = 1   # 显示碰撞体 (group 3)
    viewer._opt.geomgroup[4] = 1   # 显示地面   (group 4)

    step = 0
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # --- Isaac Lab 标准 height_scanner 观测 ---
        heights_2d, hit_z_2d = cast_ray_grid(model, data)

        # 1D 观测: shape (187,)
        obs = RAY_Z_OFFSET - heights_2d.ravel() - 0.5
        # -------------------------------------------

        # 可视化
        hit_points, heights_z = heights_to_hit_points(model, data, heights_2d.ravel())
        draw_height_dots(viewer, hit_points, heights_z)

        viewer.sync()
        time.sleep(0.001)

        # 每 100 步打印
        step += 1
        if step % 100 == 0:
            flat_min = np.argmin(obs)
            flat_max = np.argmax(obs)
            lt_min, fw_min = np.unravel_index(flat_min, heights_2d.shape)
            lt_max, fw_max = np.unravel_index(flat_max, heights_2d.shape)

            print(f"\n--- step {step} ---")
            print(f"  obs.max = {obs[flat_max]:.3f}  @ idx={flat_max}  [{grid_pos_to_desc(lt_max, fw_max)}]"
                  f"  (lt={lt_max}, fw={fw_max})")
            print(f"------------------------------\n")
