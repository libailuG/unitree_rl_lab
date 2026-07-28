'''
conda activate env_isaaclab_0

'''


'''

python scripts/tools/convert_urdf.py \
  /home/libai/00_isaaclab/unitree_rl_lab/taixi_model/A1/urdf/a1.urdf \
  /home/libai/00_isaaclab/unitree_rl_lab/taixi_model/A1/usd/a1.usd  \
  --joint-target-type position 

python scripts/tools/convert_urdf.py \
  /home/libai/00_isaaclab/unitree_rl_lab/taixi_model/A2/urdf/a2.urdf \
  /home/libai/00_isaaclab/unitree_rl_lab/taixi_model/A2/usd/a2.usd  \
  --joint-target-type position 


python scripts/tools/convert_urdf.py \
  /home/libai/00_isaaclab/unitree_rl_lab/taixi_model/A2/urdf/a2.urdf \
  /home/libai/00_isaaclab/unitree_rl_lab/taixi_model/A2/usd/a2_fix_base.usd  \
  --joint-target-type position \
  --fix-base

'''

'''

python -m pip install -e source/unitree_rl_lab

'''


'''
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --num_envs 8192 --headless



python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity \
    --load_run 2026-06-29_14-59-27 \
    --checkpoint model_34000.pt


python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity \
    --num_envs 16 \
    --checkpoint /home/libai/00_isaaclab/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-06-29_14-59-27/model_34000.pt


python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --num_envs 1 


'''

'''
python scripts/rsl_rl/train.py \
--task LeggedLab-Isaac-AMP-G1-v0 \
--num_envs 4096 \
--headless --max_iterations 50000 \
--resume \
--load_run 2026-07-03_09-23-51 \
--checkpoint model_600.pt



python scripts/rsl_rl/train.py --task Taixi-A1-Velocity --num_envs 8192 --headless

python scripts/rsl_rl/play.py --task Taixi-A1-Velocity \
    --num_envs 16 \
    --checkpoint /home/libai/00_isaaclab/unitree_rl_lab/logs/rsl_rl/taixi_a1_velocity/2026-07-02_15-49-31/model_2000.pt

python scripts/rsl_rl/train.py \
    --task Taixi-A1-Velocity \
    --num_envs 8192 \
    --headless \
    --resume \
    --load_run 2026-07-02_15-49-31 \
    --checkpoint model_2000.pt

python scripts/rsl_rl/play.py --task Taixi-A1-Velocity \
    --num_envs 16 \
    --checkpoint /home/libai/00_isaaclab/unitree_rl_lab/logs/rsl_rl/taixi_a1_velocity/2026-07-02_16-34-31/model_43000.pt

python scripts/rsl_rl/play.py --task Taixi-A1-Velocity \
    --num_envs 16 \
    --checkpoint /home/libai/00_isaaclab/unitree_rl_lab/logs/rsl_rl/taixi_a1_velocity/2026-07-07_09-04-42/model_14500.pt
  
python scripts/rsl_rl/play.py --task Taixi-A2-Velocity \
    --num_envs 16 \
    --checkpoint /home/libai/00_isaaclab/unitree_rl_lab/logs/rsl_rl/taixi_a2_velocity/2026-07-07_16-37-33/model_1600.pt
  
python scripts/rsl_rl/play.py --task Taixi-A2-Velocity \
    --num_envs 16 \
    --checkpoint /home/libai/00_isaaclab/unitree_rl_lab/logs/rsl_rl/taixi_a2_velocity/2026-07-09_11-37-05/model_4000.pt
  
    

#isaacsim
python scripts/rsl_rl/test_default_pose.py --task Taixi-A1-Velocity
python scripts/rsl_rl/test_default_pose.py --task Taixi-A2-Velocity

tensor([[ 0.2074,  0.2442,  0.1195, -0.1624, -1.4835, -1.5382, -5.7831, -5.1767,
         -0.0081, -0.4511,  0.6456,  0.2652]], device='cuda:0')

tensor([[-0.1194,  0.3893,  0.0053,  0.2237, -0.5367, -0.1111, -2.5664, -2.1492,
          3.9803,  3.1489,  0.4964, -0.0428]], device='cuda:0')

#mujoco
  act_right_roll_joint_1              +0.226 Nm
  act_right_thigh_joint_2             -0.093 Nm
  act_right_pitch_joint_3             -1.068 Nm
  act_right_pitch_joint_4             -3.397 Nm
  act_right_pitch_joint_5             +0.540 Nm
  act_right_roll_joint_6              +0.355 Nm
  act_left_roll_joint_1               +0.291 Nm
  act_left_thigh_joint_2              -0.061 Nm
  act_left_pitch_joint_3              -0.667 Nm
  act_left_pitch_joint_4              -4.337 Nm
  act_left_pitch_joint_5              +1.079 Nm
  act_left_roll_joint_6               +0.289 Nm



rsl-rl-lib 2.3.3

./isaaclab.sh -p -m pip install rsl-rl-lib==3.0.1

'''

'''

# 打印各关节应用扭矩
torque = data.actuator_force[i]


'''



# rough
'''
python scripts/rsl_rl/train.py --task Taixi-A2-Velocity-Rough --num_envs 8192 --headless

python scripts/rsl_rl/play.py --task Taixi-A2-Velocity-Rough \
    --num_envs 16 \
    --checkpoint /home/libai/05_unitree_rl_lab/unitree_rl_lab/logs/rsl_rl/taixi_a2_velocity_rough/2026-07-28_09-58-52/model_2700.pt


python scripts/rsl_rl/play.py --task Taixi-A2-Velocity-Rough \
    --num_envs 16 \
    --checkpoint /home/libai/00_isaaclab/unitree_rl_lab/logs/rsl_rl/taixi_a2_velocity_rough/2026-07-10_17-18-46/model_1000.pt


'''



'''
Unitree-G1-29dof-Velocity-Rough

python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-Rough --num_envs 8192 --headless

python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity-Rough-Play \
    --num_envs 16 \
    --checkpoint /home/libai/00_isaaclab/unitree_rl_lab/logs/rsl_rl/g1_rough/2026-07-13_10-18-18/model_2999.pt


'''




'''
Taixi-A2-Velocity-Rough


python scripts/rsl_rl/train.py --task Taixi-A2-Velocity-Rough --num_envs 8192 --headless

python scripts/rsl_rl/play.py --task Taixi-A2-Velocity-Rough \
    --num_envs 16 \
    --checkpoint /home/libai/00_isaaclab/unitree_rl_lab/logs/rsl_rl/taixi_a2_velocity_rough/2026-07-16_09-11-43/model_3000.pt



python scripts/rsl_rl/test_default_pose.py --task Taixi-A2-Velocity-Rough \
    --num_envs 1



'''


'''

python scripts/mimic/csv_to_npz.py   --input_file source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/fallAndGetUp1_subject1.csv   --output_name source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/fallAndGetUp1_subject1.npz   --input_fps 30   --output_fps 30

python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-Dance-102 --num_envs 8192 --headless

python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Mimic-Dance-102 \
    --num_envs 16 \
    --checkpoint /home/libai/00_isaaclab/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_mimic_dance_102/2026-07-21_16-44-30/model_29999.pt

    

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python scripts/mimic/replay_npz.py  -f  /home/libai/00_isaaclab/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/fallAndGetUp1_subject1.npz





python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Mimic-Dance-102 \
    --num_envs 8192 \
    --headless \
    --resume \
    --load_run 2026-07-21_16-44-30 \
    --checkpoint model_29999.pt


'''





'''

python scripts/rsl_rl/play.py --help

usage: play.py [-h] [--video] [--video_length VIDEO_LENGTH] [--disable_fabric] [--num_envs NUM_ENVS] [--task TASK] [--use_pretrained_checkpoint] [--real-time] [--experiment_name EXPERIMENT_NAME] [--run_name RUN_NAME] [--resume] [--load_run LOAD_RUN]
               [--checkpoint CHECKPOINT] [--logger {neptune,wandb,tensorboard}] [--log_project_name LOG_PROJECT_NAME] [--headless] [--livestream {0,1,2}] [--enable_cameras] [--xr] [--device DEVICE] [--verbose] [--info] [--experience EXPERIENCE]
               [--rendering_mode {balanced,performance,quality}] [--kit_args KIT_ARGS] [--anim_recording_enabled] [--anim_recording_start_time ANIM_RECORDING_START_TIME] [--anim_recording_stop_time ANIM_RECORDING_STOP_TIME]

Train an RL agent with RSL-RL.

options:
  -h, --help            show this help message and exit
  --video               Record videos during training.
  --video_length VIDEO_LENGTH
                        Length of the recorded video (in steps).
  --disable_fabric      Disable fabric and use USD I/O operations.
  --num_envs NUM_ENVS   Number of environments to simulate.
  --task TASK           Name of the task.
  --use_pretrained_checkpoint
                        Use the pre-trained checkpoint from Nucleus.
  --real-time           Run in real-time, if possible.

rsl_rl:
  Arguments for RSL-RL agent.

  --experiment_name EXPERIMENT_NAME
                        Name of the experiment folder where logs will be stored.
  --run_name RUN_NAME   Run name suffix to the log directory.
  --resume              Whether to resume from a checkpoint.
  --load_run LOAD_RUN   Name of the run folder to resume from.
  --checkpoint CHECKPOINT
                        Checkpoint file to resume from.
  --logger {neptune,wandb,tensorboard}
                        Logger module to use.
  --log_project_name LOG_PROJECT_NAME
                        Name of the logging project when using wandb or neptune.

app_launcher arguments:
  Arguments for the AppLauncher. For more details, please check the documentation.

  --headless            Force display off at all times.
  --livestream {0,1,2}  Force enable livestreaming. Mapping corresponds to that for the `LIVESTREAM` environment variable.
  --enable_cameras      Enable camera sensors and relevant extension dependencies.
  --xr                  Enable XR mode for VR/AR applications.
  --device DEVICE       The device to run the simulation on. Can be "cpu", "cuda", "cuda:N", where N is the device ID
  --verbose             Enable verbose-level log output from the SimulationApp.
  --info                Enable info-level log output from the SimulationApp.
  --experience EXPERIENCE
                        The experience file to load when launching the SimulationApp. If an empty string is provided, the experience file is determined based on the headless flag. If a relative path is provided, it is resolved relative to the `apps` folder in Isaac Sim
                        and Isaac Lab (in that order).
  --rendering_mode {balanced,performance,quality}
                        Sets the rendering mode. Preset settings files can be found in apps/rendering_modes. Can be "performance", "balanced", or "quality". Individual settings can be overwritten by using the RenderCfg class.
  --kit_args KIT_ARGS   Command line arguments for Omniverse Kit as a string separated by a space delimiter. Example usage: --kit_args "--ext-folder=/path/to/ext1 --ext-folder=/path/to/ext2"
  --anim_recording_enabled
                        Enable recording time-sampled USD animations from IsaacLab PhysX simulations.
  --anim_recording_start_time ANIM_RECORDING_START_TIME
                        Set time that animation recording begins playing. If not set, the recording will start from the beginning.
  --anim_recording_stop_time ANIM_RECORDING_STOP_TIME
                        Set time that animation recording stops playing. If the process is shutdown before the stop time is exceeded, then the animation is not recorded.


'''


'''

python scripts/rsl_rl/train.py --help
usage: train.py [-h] [--video] [--video_length VIDEO_LENGTH] [--video_interval VIDEO_INTERVAL] [--num_envs NUM_ENVS]
                [--task {Taixi-A1-Velocity,Unitree-G1-29dof-Velocity,Unitree-Go2-Velocity,Unitree-H1-Velocity,Unitree-G1-29dof-Mimic-Dance-102,Unitree-G1-29dof-Mimic-Gangnanm-Style}] [--seed SEED] [--max_iterations MAX_ITERATIONS]
                [--distributed] [--experiment_name EXPERIMENT_NAME] [--run_name RUN_NAME] [--resume] [--load_run LOAD_RUN] [--checkpoint CHECKPOINT] [--logger {wandb,neptune,tensorboard}] [--log_project_name LOG_PROJECT_NAME]
                [--headless] [--livestream {0,1,2}] [--enable_cameras] [--xr] [--device DEVICE] [--verbose] [--info] [--experience EXPERIENCE] [--rendering_mode {performance,balanced,quality}] [--kit_args KIT_ARGS]
                [--anim_recording_enabled] [--anim_recording_start_time ANIM_RECORDING_START_TIME] [--anim_recording_stop_time ANIM_RECORDING_STOP_TIME]

Train an RL agent with RSL-RL.

options:
  -h, --help            show this help message and exit
  --video               Record videos during training.
  --video_length VIDEO_LENGTH
                        Length of the recorded video (in steps).
  --video_interval VIDEO_INTERVAL
                        Interval between video recordings (in steps).
  --num_envs NUM_ENVS   Number of environments to simulate.
  --task {Taixi-A1-Velocity,Unitree-G1-29dof-Velocity,Unitree-Go2-Velocity,Unitree-H1-Velocity,Unitree-G1-29dof-Mimic-Dance-102,Unitree-G1-29dof-Mimic-Gangnanm-Style}
                        Name of the task.
  --seed SEED           Seed used for the environment
  --max_iterations MAX_ITERATIONS
                        RL Policy training iterations.
  --distributed         Run training with multiple GPUs or nodes.

rsl_rl:
  Arguments for RSL-RL agent.

  --experiment_name EXPERIMENT_NAME
                        Name of the experiment folder where logs will be stored.
  --run_name RUN_NAME   Run name suffix to the log directory.
  --resume              Whether to resume from a checkpoint.
  --load_run LOAD_RUN   Name of the run folder to resume from.
  --checkpoint CHECKPOINT
                        Checkpoint file to resume from.
  --logger {wandb,neptune,tensorboard}
                        Logger module to use.
  --log_project_name LOG_PROJECT_NAME
                        Name of the logging project when using wandb or neptune.

app_launcher arguments:
  Arguments for the AppLauncher. For more details, please check the documentation.

  --headless            Force display off at all times.
  --livestream {0,1,2}  Force enable livestreaming. Mapping corresponds to that for the `LIVESTREAM` environment variable.
  --enable_cameras      Enable camera sensors and relevant extension dependencies.
  --xr                  Enable XR mode for VR/AR applications.
  --device DEVICE       The device to run the simulation on. Can be "cpu", "cuda", "cuda:N", where N is the device ID
  --verbose             Enable verbose-level log output from the SimulationApp.
  --info                Enable info-level log output from the SimulationApp.
  --experience EXPERIENCE
                        The experience file to load when launching the SimulationApp. If an empty string is provided, the experience file is determined based on the headless flag. If a relative path is provided, it is resolved relative
                        to the `apps` folder in Isaac Sim and Isaac Lab (in that order).
  --rendering_mode {performance,balanced,quality}
                        Sets the rendering mode. Preset settings files can be found in apps/rendering_modes. Can be "performance", "balanced", or "quality". Individual settings can be overwritten by using the RenderCfg class.
  --kit_args KIT_ARGS   Command line arguments for Omniverse Kit as a string separated by a space delimiter. Example usage: --kit_args "--ext-folder=/path/to/ext1 --ext-folder=/path/to/ext2"
  --anim_recording_enabled
                        Enable recording time-sampled USD animations from IsaacLab PhysX simulations.
  --anim_recording_start_time ANIM_RECORDING_START_TIME
                        Set time that animation recording begins playing. If not set, the recording will start from the beginning.
  --anim_recording_stop_time ANIM_RECORDING_STOP_TIME
                        Set time that animation recording stops playing. If the process is shutdown before the stop time is exceeded, then the animation is not recorded.


'''

















