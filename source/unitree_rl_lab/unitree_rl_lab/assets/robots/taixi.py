# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
import math 

from . import taixi_actuators

TAIXI_MODEL_DIR = "/home/libai/05_unitree_rl_lab/unitree_rl_lab/taixi_model"  # Replace with the actual path to your unitree_model directory


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


@configclass
class UnitreeUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )

    def replace_asset(self, meshes_dir, urdf_path):
        """Replace the asset with a temporary copy to avoid modifying the original asset.

        When need to change the collisions, place the modified URDF file separately in this repository,
        and let `meshes_dir` be provided by `unitree_ros`.
        This function will auto construct a complete `robot_description` file structure in the `/tmp` directory.
        Note: The mesh references inside the URDF should be in the same directory level as the URDF itself.
        """
        tmp_meshes_dir = "/tmp/IsaacLab/unitree_rl_lab/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/unitree_rl_lab", exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = "/tmp/IsaacLab/unitree_rl_lab/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)


""" Configuration for the Unitree robots."""

TAIXI_A1_CFG = UnitreeArticulationCfg(

    spawn=UnitreeUsdFileCfg(
        usd_path=f"{TAIXI_MODEL_DIR}/A1/usd/a1.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.97),
        joint_pos={
            ".*roll_joint_1" : 0.0,
            ".*thigh_joint_2" : 0.0,
            ".*pitch_joint_3" : -20.0/180.0*math.pi,
            ".*pitch_joint_4" : 50.0/180.0*math.pi,
            ".*pitch_joint_5" : -30.0/180.0*math.pi,
            ".*roll_joint_6" : 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    # actuators={
    #     "a1": taixi_actuators.UnitreeActuatorCfg_A1(
    #         joint_names_expr=[".*"],
    #         stiffness=20.0,
    #         damping=2.0,
    #         friction=0.01,
    #     ),
    # },
    actuators={
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*roll_joint_1",".*thigh_joint_2",".*pitch_joint_3",".*pitch_joint_4",".*pitch_joint_5",".*roll_joint_6"], 
            effort_limit_sim=200,
            velocity_limit_sim=32.0,
            stiffness={
                ".*roll_joint_1": 50.0,
                ".*thigh_joint_2": 50.0,
                ".*pitch_joint_3": 50.0,
                ".*pitch_joint_4": 50.0,
                ".*pitch_joint_5": 50.0,
                ".*roll_joint_6": 50.0,
            },
            damping={
                ".*roll_joint_1": 0.3,
                ".*thigh_joint_2": 0.3,
                ".*pitch_joint_3": 0.3,
                ".*pitch_joint_4": 0.3,
                ".*pitch_joint_5": 0.3,
                ".*roll_joint_6": 0.3,
            },
            armature=0.001,
        ),
    },
    joint_sdk_names=[
        "right_roll_joint_1",
        "right_thigh_joint_2",
        "right_pitch_joint_3",
        "right_pitch_joint_4",
        "right_pitch_joint_5",
        "right_roll_joint_6",
        "left_roll_joint_1",
        "left_thigh_joint_2",
        "left_pitch_joint_3",
        "left_pitch_joint_4",
        "left_pitch_joint_5",
        "left_roll_joint_6",
    ],
)


TAIXI_A2_CFG = UnitreeArticulationCfg(

    spawn=UnitreeUsdFileCfg(
        usd_path=f"{TAIXI_MODEL_DIR}/A2/usd/a2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.02),
        joint_pos={
            ".*hip_roll.*" : 0.0,
            ".*hip_yaw.*" : 0.0,
            ".*hip_pitch.*" : -10.0/180.0*math.pi,
            ".*knee.*" : 20.0/180.0*math.pi,
            ".*ankle_pitch.*" : -10.0/180.0*math.pi,
            ".*ankle_roll.*" : 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    # actuators={
    #     "a1": taixi_actuators.UnitreeActuatorCfg_A1(
    #         joint_names_expr=[".*"],
    #         stiffness=20.0,
    #         damping=2.0,
    #         friction=0.01,
    #     ),
    # },
    actuators={
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_roll.*",".*hip_yaw.*",".*hip_pitch.*",".*knee.*",".*ankle_pitch.*",".*ankle_roll.*"], 
            effort_limit_sim=200,
            velocity_limit_sim=32.0,
            stiffness={
                ".*hip_roll.*": 50.0,
                ".*hip_yaw.*": 50.0,
                ".*hip_pitch.*": 50.0,
                ".*knee.*": 50.0,
                ".*ankle_pitch.*": 50.0,
                ".*ankle_roll.*": 50.0,
            },
            damping={
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.3,
                ".*hip_pitch.*": 0.3,
                ".*knee.*": 0.3,
                ".*ankle_pitch.*": 0.3,
                ".*ankle_roll.*": 0.3,
            },
            armature=0.000,
        ),
    },
    joint_sdk_names=[
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",    
        "left_ankle_roll_joint",
    ],
)


TAIXI_A2_ROUGH_CFG = UnitreeArticulationCfg(

    spawn=UnitreeUsdFileCfg(
        usd_path=f"{TAIXI_MODEL_DIR}/A2/usd/a2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 1.05),
        pos=(0.2, 0.6, 1.05),
        rot=(0.9659, 0.0, 0.0, 0.2588),
        joint_pos={
            ".*hip_roll.*" : 0.0,
            ".*hip_yaw.*" : 0.0,
            ".*hip_pitch.*" : -10.0/180.0*math.pi,
            ".*knee.*" : 20.0/180.0*math.pi,
            ".*ankle_pitch.*" : -10.0/180.0*math.pi,
            ".*ankle_roll.*" : 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    # actuators={
    #     "a1": taixi_actuators.UnitreeActuatorCfg_A1(
    #         joint_names_expr=[".*"],
    #         stiffness=20.0,
    #         damping=2.0,
    #         friction=0.01,
    #     ),
    # },
    actuators={
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_roll.*",".*hip_yaw.*",".*hip_pitch.*",".*knee.*",".*ankle_pitch.*",".*ankle_roll.*"], 
            effort_limit_sim=200,
            velocity_limit_sim=32.0,
            stiffness={
                ".*hip_roll.*": 50.0,
                ".*hip_yaw.*": 50.0,
                ".*hip_pitch.*": 50.0,
                ".*knee.*": 50.0,
                ".*ankle_pitch.*": 50.0,
                ".*ankle_roll.*": 50.0,
            },
            damping={
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.3,
                ".*hip_pitch.*": 0.3,
                ".*knee.*": 0.3,
                ".*ankle_pitch.*": 0.3,
                ".*ankle_roll.*": 0.3,
            },
            armature=0.000,
        ),
    },
    joint_sdk_names=[
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",    
        "left_ankle_roll_joint",
    ],
)


TAIXI_A2_FIX_BASE_CFG = UnitreeArticulationCfg(

    spawn=UnitreeUsdFileCfg(
        usd_path=f"{TAIXI_MODEL_DIR}/A2/usd/a2_fix_base.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.2),
        joint_pos={
            ".*hip_roll.*" : 0.0,
            ".*hip_yaw.*" : 0.0,
            ".*hip_pitch.*" : -10.0/180.0*math.pi,
            ".*knee.*" : 20.0/180.0*math.pi,
            ".*ankle_pitch.*" : -10.0/180.0*math.pi,
            ".*ankle_roll.*" : 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    # actuators={
    #     "a1": taixi_actuators.UnitreeActuatorCfg_A1(
    #         joint_names_expr=[".*"],
    #         stiffness=20.0,
    #         damping=2.0,
    #         friction=0.01,
    #     ),
    # },
    actuators={
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_roll.*",".*hip_yaw.*",".*hip_pitch.*",".*knee.*",".*ankle_pitch.*",".*ankle_roll.*"], 
            effort_limit_sim=200,
            velocity_limit_sim=32.0,
            stiffness={
                ".*hip_roll.*": 50.0,
                ".*hip_yaw.*": 50.0,
                ".*hip_pitch.*": 50.0,
                ".*knee.*": 50.0,
                ".*ankle_pitch.*": 50.0,
                ".*ankle_roll.*": 50.0,
            },
            damping={
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.3,
                ".*hip_pitch.*": 0.3,
                ".*knee.*": 0.3,
                ".*ankle_pitch.*": 0.3,
                ".*ankle_roll.*": 0.3,
            },
            armature=0.000,
        ),
    },
    joint_sdk_names=[
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",    
        "left_ankle_roll_joint",
    ],
)