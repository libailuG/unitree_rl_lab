# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play with default joint positions (zero actions)."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with default joint positions (zero actions).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Taixi-A2-Velocity-Rough", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with default joint positions (zero actions)."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    # ── 添加地形长方体 (与 a2_box.xml 对齐) ──
    # a2_box.xml:
    #   robot base_link @ world (0, 0, 1.0)
    #   box geom @ world (-1.5, 0.4, 0.05), MuJoCo half-extents "0.2 0.2 0.1"
    #   → box 相对于 robot 的偏移: (-1.5, 0.4, -0.95)
    # Isaac Sim:
    #   robot init_pos = (0, 0, 1.05)  (TAIXI_A2_ROUGH_CFG)
    #   → box world pos = robot_init + relative_offset = (-1.5, 0.4, 0.10)

    box_pos = (
        -16.0 - 1.0,
        -16.0 + 0.7,
        0.1,     # = 1.05 + (-0.95) = 0.10
    )
    # prim_path 放在 /World/ground 下，RayCaster 扫描 terrain 时自动覆盖
    env_cfg.scene.terrain_box = RigidObjectCfg(
        prim_path="/World/ground/terrain_box",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.2),  # 全尺寸 (MuJoCo half-extents × 2)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=box_pos),
        collision_group=-1,
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "videos", "play_default_joint_pos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    dt = env.unwrapped.step_dt

    # height_scanner grid: 17×11 = 187 rays, "xy" ordering
    # Four corners (x from -0.8 to 0.8, y from -0.5 to 0.5):
    #   index 0:   rear-right,  index 16:  front-right
    #   index 170: rear-left,   index 186: front-left

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # zero actions -> robot stays at default joint positions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # env stepping
            obs, _, _, _, _ = env.step(actions)

        # compute and print height_scan four corner values every 100 steps
        # height_scan = ObsTerm(
        #     func=mdp_2.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),  # only applied when corruption enabled
        #     clip=(-1.0, 1.0),
        # )
        # Formula: height_scan = sensor_z - hit_z - offset(0.5), then clamp(-1, 1)
        if timestep % 100 == 0:
            # 打印机器人世界位置
            root_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]  # (x, y, z) env 0
            print(f"[Step {timestep}] robot world pos = ({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f})")

            hs_data = env.unwrapped.scene["height_scanner"].data  # type: ignore[attr-defined]
            ray_hits = hs_data.ray_hits_w
            if ray_hits is None:
                print(f"[Step {timestep}] height_scanner ray_hits_w is None — sensor not updated yet")
            else:
                num_rays = ray_hits.shape[1]
                # dynamically compute corner indices from actual grid size
                nx = int(1.6 / 0.1) + 1  # 17
                ny = int(1.0 / 0.1) + 1  # 11
                if num_rays == nx * ny:
                    corner_idx = [0, nx - 1, (ny - 1) * nx, ny * nx - 1]  # [0, 16, 170, 186]
                else:
                    corner_idx = [0, num_rays - 1, num_rays // 2 - 1, num_rays // 2]
                # height_scan formula: sensor_z - hit_z - 0.5, clamped to (-1, 1)
                sensor_z = hs_data.pos_w[:, 2]                          # (num_envs,)
                height_scan_full = sensor_z.unsqueeze(1) - ray_hits[..., 2] - 0.5  # (num_envs, 187)
                height_scan_full = torch.clamp(height_scan_full, -1.0, 1.0)
                print(corner_idx)
                corner_vals = height_scan_full[:, corner_idx]           # (num_envs, 4)
                print(f"[Step {timestep}] height_scan({num_rays} rays), clip=(-1,1):"
                      f"  rear-right={corner_vals[0, 0]:.4f}  front-right={corner_vals[0, 1]:.4f}"
                      f"  rear-left={corner_vals[0, 2]:.4f}  front-left={corner_vals[0, 3]:.4f}")

        timestep += 1

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
