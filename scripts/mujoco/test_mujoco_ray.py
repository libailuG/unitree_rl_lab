import mujoco
import mujoco.viewer
import numpy as np
import time
import math

# 加载模型
MODEL_PATH = "/home/libai/00_isaaclab/unitree_rl_lab/taixi_model/A2/urdf/a2_ray.xml"
# MODEL_PATH = '/home/libai/00_isaaclab/taixi_centaur01/model/taixi_centaur01/urdf_foot/taixi_centaur01.xml'

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)




def draw_red_dots(viewer, points, radius=0.035):
    """Draw sampled ray hit points into the viewer's user scene."""
    scene = viewer.user_scn
    scene.ngeom = 0

    dot_size = np.array([radius, radius, radius], dtype=np.float64)
    dot_color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    dot_mat = np.eye(3, dtype=np.float64).reshape(-1)

    max_points = min(len(points), scene.maxgeom)
    for point in points[:max_points]:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            dot_size,
            point,
            dot_mat,
            dot_color,
        )
        scene.ngeom += 1


def get_rangefinder_hits(model, data):
    """Read XML-defined rangefinder sensors and return world-space hit positions.

    Unlike ``mujoco.mj_ray``, sensor values are computed automatically during
    ``mj_step`` — no manual raycast loop needed.
    """
    hit_points = []
    for sensor_id in range(model.nsensor):
        if model.sensor_type[sensor_id] != mujoco.mjtSensor.mjSENS_RANGEFINDER:
            continue
        distance = data.sensordata[sensor_id]
        if distance < 0:  # no hit (ray missed everything)
            continue

        # Site whose Z-axis defines the ray direction.
        site_id = model.sensor_objid[sensor_id]
        site_pos = data.site_xpos[site_id]

        # Site Z-axis in world frame — 3rd column of the 3×3 rotation matrix
        # (column-major, FORTRAN order).  site_xmat is (nsite, 9).
        site_z = data.site_xmat[site_id][6:9]

        hit_point = site_pos + distance * site_z
        hit_points.append(hit_point)

    return np.asarray(hit_points, dtype=np.float64)





# 仿真循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    # viewer.vopt.geomgroup[3] = 1  # 显示碰撞体
    # viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1  # 显示执行器力

    while viewer.is_running():
        # 持续下发位置指令 (PD 控制器会自动跟踪)
        # data.ctrl[:] = stand_pose

        mujoco.mj_step(model, data)

        hit_points = get_rangefinder_hits(model, data)
        draw_red_dots(viewer, hit_points)

        viewer.sync()
        time.sleep(0.001)
