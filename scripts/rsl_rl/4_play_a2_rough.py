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
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# exported policy path
policy_path = os.path.join(
    os.getcwd(), "logs/rsl_rl/", "taixi_a2_velocity_rough/2026-07-28_09-58-52/",
    "exported/", "policy_260728.pt"
)

# output path for step data
output_dir = os.path.join(os.getcwd(), "logs", "step_data")
os.makedirs(output_dir, exist_ok=True)

# ── Observation structure ──
# Each obs term: (name, dim, description)
OBS_TERMS = [
    ("base_lin_vel",      3, "linear velocity in body frame (vx, vy, vz)"),
    ("base_ang_vel",      3, "angular velocity in body frame (wx, wy, wz)"),
    ("projected_gravity", 3, "gravity vector in body frame (gx, gy, gz)"),
    ("velocity_commands", 3, "commanded base velocity (vx_cmd, vy_cmd, wz_cmd)"),
    ("joint_pos",        12, "relative joint positions (12 DoF, rad)"),
    ("joint_vel",        12, "relative joint velocities (12 DoF, rad/s)"),
    ("actions",          12, "last action applied (12 DoF)"),
    ("height_scan",     187, "height scanner rays (17×11 grid, clip -1 to 1)"),
    ("gait_phase",        2, "gait phase (sin, cos), period=0.8s"),
]
OBS_DIM = sum(d for _, d, _ in OBS_TERMS)  # 237

POLICY_HISTORY  = 3
CRITIC_HISTORY  = 10

# Precompute offset of each term within a single history frame
_term_offset = {}
_off = 0
for _name, _dim, _ in OBS_TERMS:
    _term_offset[_name] = _off
    _off += _dim


def get_term_hist0(policy_obs_flat: torch.Tensor, history: int, term_name: str) -> torch.Tensor:
    """Extract hist_0 of a named term from a flat policy observation.

    Layout: for each term, all dims of hist_0 come first, then hist_1, etc.
      [dim0_h0, dim1_h0, ..., dim{D-1}_h0, dim0_h1, ..., dim{D-1}_h{H-1}]
    """
    dim = next(d for n, d, _ in OBS_TERMS if n == term_name)
    start = _term_offset[term_name] * history
    return policy_obs_flat[start : start + dim]


def write_obs(lines: list[str], tensor_1d: torch.Tensor, prefix: str, history: int):
    """Split a flat 1-D obs tensor into named components and format.

    Layout: for each term, all `dim` values of history frame 0 come first,
    then all `dim` values of history frame 1, etc.
    E.g. joint_pos (dim=12, H=3):
      [j0_h0, j1_h0, ..., j11_h0, j0_h1, ..., j11_h1, j0_h2, ..., j11_h2]

    Args:
        lines: output line list (mutated in-place).
        tensor_1d: flat observation tensor, shape (obs_dim * history,).
        prefix: variable name prefix, e.g. "step0_obs_policy".
        history: number of history frames.
    """
    t = tensor_1d.detach().cpu().tolist()
    expected = OBS_DIM * history
    assert len(t) == expected, f"Expected {expected}, got {len(t)}"

    offset = 0
    for name, dim, desc in OBS_TERMS:
        # Each term occupies history * dim values:
        #   [all_dim_h0, all_dim_h1, ..., all_dim_h{H-1}]
        # Reshape to (history, dim) rows: each row = one history frame
        chunk = t[offset : offset + history * dim]
        rows = [chunk[h * dim : (h + 1) * dim] for h in range(history)]
        offset += history * dim

        lines.append(f"# {name}: {desc}")
        lines.append(f"{prefix}_{name} = [")
        for i, row in enumerate(rows):
            formatted = ", ".join(f"{v: 11.5f}" for v in row)
            comma = "," if i < len(rows) - 1 else ","
            tag = f"  # hist_{i}" if history > 1 else ""
            lines.append(f"    [{formatted}]{comma}{tag}")
        lines.append(f"]  # ({history} × {dim})")
        lines.append("")


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

    # ── Warmup: wait until joint_pos starts fluctuating ──
    obs_dict, _ = env.reset()
    init_joint_pos = get_term_hist0(obs_dict["policy"][0], POLICY_HISTORY, "joint_pos").clone()
    print(f"[INFO]: Initial joint_pos (hist_0): {init_joint_pos.tolist()}")

    warmup_steps = 0
    max_warmup = 500
    threshold = 0.01  # rad — consider "started moving" when any joint changes by >0.01 rad

    while warmup_steps < max_warmup:
        with torch.inference_mode():
            actions = policy(obs_dict["policy"])
            obs_dict, _, _, _, _ = env.step(actions)
        warmup_steps += 1

        curr_joint_pos = get_term_hist0(obs_dict["policy"][0], POLICY_HISTORY, "joint_pos")
        max_diff = (curr_joint_pos - init_joint_pos).abs().max().item()
        if max_diff > threshold:
            print(f"[INFO]: joint_pos started fluctuating at warmup step {warmup_steps} "
                  f"(max diff from init = {max_diff:.5f} rad)")
            break
    else:
        print(f"[WARN]: joint_pos did not fluctuate within {max_warmup} steps, recording anyway")

    # ── Record 3 steps ──
    all_lines = []
    all_lines.append("# Auto-generated step data from 4_play_a2_rough.py")
    all_lines.append(f"# task: {args_cli.task}, num_envs: {args_cli.num_envs}")
    all_lines.append(f"# warmup steps: {warmup_steps}, fluctuation_threshold: {threshold} rad")
    all_lines.append(f"# obs_dim per history step: {OBS_DIM}")
    all_lines.append(f"# policy history: {POLICY_HISTORY} → {OBS_DIM * POLICY_HISTORY} dims")
    all_lines.append(f"# critic history: {CRITIC_HISTORY} → {OBS_DIM * CRITIC_HISTORY} dims")
    all_lines.append("")

    for step in range(0, 4):
        if step == 0:
            all_lines.append("# ============================================================")
            all_lines.append("#  Step 0 (first after fluctuation detected)")
            all_lines.append("# ============================================================")
        else:
            all_lines.append("# ============================================================")
            all_lines.append(f"#  Step {step}")
            all_lines.append("# ============================================================")

        # Policy observation (flat → structured)
        obs_policy_flat = obs_dict["policy"][0]  # shape (1, N) → (N,)
        all_lines.append(f"# --- policy observation ({POLICY_HISTORY} history × {OBS_DIM} = {OBS_DIM * POLICY_HISTORY}) ---")
        write_obs(all_lines, obs_policy_flat, f"step{step}_obs_policy", POLICY_HISTORY)

        # Critic observation (flat → structured)
        obs_critic_flat = obs_dict["critic"][0]  # shape (1, N) → (N,)
        all_lines.append(f"# --- critic observation ({CRITIC_HISTORY} history × {OBS_DIM} = {OBS_DIM * CRITIC_HISTORY}) ---")
        write_obs(all_lines, obs_critic_flat, f"step{step}_obs_critic", CRITIC_HISTORY)

        # Step forward (only for steps 0-2, to record steps 1-3)
        if step < 3:
            with torch.inference_mode():
                actions = policy(obs_dict["policy"])
                obs_dict, _, _, _, _ = env.step(actions)

    # ── Write output ──
    output_path = os.path.join(output_dir, "step_data.py")
    with open(output_path, "w") as f:
        f.write("\n".join(all_lines))
    print(f"[INFO]: Step data saved to: {output_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
