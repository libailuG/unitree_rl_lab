# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to display a robot in Isaac Sim at its default position."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Display a robot in Isaac Sim at its default position.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.taixi import TAIXI_A2_ROUGH_CFG as ROBOT_CFG


@configclass
class RobotDisplaySceneCfg(InteractiveSceneCfg):
    """Configuration for a simple robot display scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    """Display a robot in Isaac Sim."""
    # create simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # create scene
    scene_cfg = RobotDisplaySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # reset simulation
    sim.reset()

    print("[INFO] Robot spawned at default position.")
    print(f"[INFO] Default position: {ROBOT_CFG.init_state.pos}")
    print("[INFO] Simulation running... Press Ctrl+C or close the window to exit.")

    # get the robot articulation
    robot: Articulation = scene["robot"]

    # set default root state once
    robot.write_root_state_to_sim(robot.data.default_root_state)

    # fall detection threshold (half of default base height)
    fall_height = robot.data.default_root_state[0, 2].item() * 0.5

    # simulation loop
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        # set PD controller targets to default joint positions
        robot.set_joint_position_target(robot.data.default_joint_pos)
        # write to simulation
        scene.write_data_to_sim()
        # step physics (PD controller holds default pose against gravity)
        sim.step(render=True)
        # update scene buffers
        scene.update(sim_dt)

        # check for fall and reset
        root_z = robot.data.root_pos_w[0, 2].item()
        if root_z < fall_height:
            print(f"[INFO] Robot fallen (height={root_z:.2f} < {fall_height:.2f}), resetting to default pose...")
            robot.write_root_state_to_sim(robot.data.default_root_state)
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos,
                robot.data.default_joint_vel,
            )
            scene.write_data_to_sim()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
