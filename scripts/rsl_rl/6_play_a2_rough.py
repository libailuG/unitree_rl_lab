# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play an exported JIT policy on Taixi-A2-Velocity-Rough."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent on Taixi-A2-Velocity-Rough.")
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

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from libai_keyboard import TermiosKeyMonitor
torch.set_printoptions(precision=5, sci_mode=False)

# exported policy path
policy_path = os.path.join(
    os.getcwd(), "logs/rsl_rl/", "taixi_a2_velocity_rough/2026-07-28_09-58-52/",
    "exported/", "policy_260728.pt"
)


def main():
    """Play with exported JIT policy on Taixi-A2-Velocity-Rough."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "videos", "play_a2_rough"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # load exported JIT policy
    print(f"[INFO]: Loading exported policy from: {policy_path}")
    policy = torch.jit.load(policy_path)
    policy.eval()
    policy = policy.to(env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # reset environment
    obs_dict, _ = env.reset()
    timestep = 0


    monitor = TermiosKeyMonitor()

    monitor.start()

    step_flag = False
    step_count = 0
    init_joint_pos_isaaclab = torch.tensor([
        0.0,
        0.0,
        0.0,
        0.0,
        -10.0 / 180.0 * torch.pi,
        -10.0 / 180.0 * torch.pi,
        20.0 / 180.0 * torch.pi,
        20.0 / 180.0 * torch.pi,
        -10.0 / 180.0 * torch.pi,
        -10.0 / 180.0 * torch.pi,
        0.0,
        0.0,
    ])


    try:
        # simulate environment
        while simulation_app.is_running():

            monitor.frame()
            if monitor.was_just_pressed('s'):
                step_flag = True
            if monitor.was_just_pressed('q'):
                break

            if step_flag:

                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # extract policy observation from dict and run inference
                    actions = policy(obs_dict["policy"])


                    if step_count < 50:
                        actions = torch.zeros_like(actions)
                    else:
                        step_flag = False
                        print(step_count)
                        print(obs_dict["policy"])
                        print((actions[0] * 0.25 + init_joint_pos_isaaclab.to(actions.device))/torch.pi * 180.0)

                    # env stepping
                    obs_dict, _, _, _, _ = env.step(actions)
                if args_cli.video:
                    timestep += 1
                    # Exit the play loop after recording one video
                    if timestep == args_cli.video_length:
                        break

                # time delay for real-time evaluation
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

                step_count += 1

    finally:
        monitor.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
