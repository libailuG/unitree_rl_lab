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

# ============================================================
# 1. 加载插件
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

# ============================================================
# 2. 加载模型和数据
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../taixi_model/A2/urdf/a2_box_plugin.xml")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ============================================================
# 3. 解析传感器元数据
# ============================================================
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
sensor_link_name = "raycaster_link"
h_rays, v_rays, pairs = get_ray_caster_info(model, data, sensor_name)
print(f"=== Sensor: {sensor_name} ===")
print(f"h_rays={h_rays}, v_rays={v_rays}")
for i, (offset, size) in enumerate(pairs):
    print(f"  block[{i}]: offset={offset}, size={size}")
print("仿真运行中，每 100 步输出一次数据...\n")




def get_ray_caster_obs(model, data, sensor_name,sensor_link_name):
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



# ============================================================
# 4. 步进计数器
# ============================================================
step_count = 0


# ============================================================
# 5. 仿真步回调（mj_step 每步自动调用）
# ============================================================

def step_callback(_model, _data):
    global step_count, sensor_name, sensor_link_name
    step_count += 1

    # ---------- 每 100 步打印一次传感器数据 ----------
    if step_count % 100 != 0:
        return

    valid_z = get_ray_caster_obs(_model, _data, sensor_name, sensor_link_name)
    if len(valid_z) > 0:
        print(f"[step {step_count:4d}]")
        print(f"valid_z.shape={valid_z.shape},valid_z:{valid_z[0 * 17 + 0],valid_z[0 * 17 + 16],valid_z[10 * 17 + 0],valid_z[10 * 17 + 16]}")





# ============================================================
# 6. 注册回调 + 启动仿真窗口（内部是无限循环，直到关闭窗口）
# ============================================================
mujoco.set_mjcb_passive(step_callback)
mujoco.viewer.launch(model, data)
# mujoco.viewer.launch(model, data)._opt.geomgroup[3] = 1
# mujoco.viewer.launch_passive(model, data) as viewer
