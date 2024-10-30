import argparse
import numpy as np
from omni.isaac.lab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play the RL agent with RSL-RL & Teleoperation.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default=None, help="Device for interacting with environment")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab.devices import Se2Gamepad, Se2Keyboard
from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
import IsaacRay.tasks  # noqa: F401


def main():
    if args_cli.task is None:
        raise ValueError("Task name is required to play the agent.")
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir,filename="policy.pt")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir,filename="policy.onnx")

    # env.reset()
    obs, _ = env.get_observations()
    timestep = 0

    teleop_interface = None
    if args_cli.teleop_device is not None:
        if args_cli.teleop_device.lower() == "gamepad":
            teleop_interface = Se2Gamepad()
        elif args_cli.teleop_device.lower() == "keyboard":
            teleop_interface = Se2Keyboard()
        else:
            raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")
        print("Teleoperation device: ", teleop_interface)
        teleop_interface.reset()
    else:
        print("No teleoperation device specified.")


    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if args_cli.teleop_device is not None:
                target = teleop_interface.advance()

                # print(target)
            obs[:,9:12] = torch.tensor(target,dtype=torch.float32)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()