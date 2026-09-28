"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=24000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of iterations to train.")
parser.add_argument("--save_interval", type=int, default=None, help="The number of iterations between saves")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument("--reset_optimizer", action="store_true", help="Resume network weights with fresh optimizers.")
parser.add_argument("--jump_trace_path", default=None, help="New CSV path for pre-reset jump diagnostics.")
parser.add_argument("--jump_trace_envs", type=int, default=4, help="Number of jump environments to trace.")
parser.add_argument("--jump_min_thrust_time", type=float, default=None)
parser.add_argument("--jump_min_release_length", type=float, default=None)
parser.add_argument("--jump_flight_retract_length", type=float, default=None)
parser.add_argument("--jump_assisted", action="store_true", help="Opt into the historical reference controller.")
parser.add_argument("--jump_action_rate_weight", type=float, default=None)
parser.add_argument("--jump_action_smooth_weight", type=float, default=None)
parser.add_argument("--height_range", type=float, nargs=2, default=None,
                    metavar=("MIN_M", "MAX_M"), help="Height-task command range within 0.22–0.26 m.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

# from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, RslRlVecEnvWrapper


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    cli_args.configure_getup(env_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    if args_cli.height_range is not None:
        if not hasattr(env_cfg.commands, "base_height"):
            raise ValueError("--height_range requires the Motor35 height task")
        env_cfg.commands.base_height.height_range = tuple(args_cli.height_range)
    if args_cli.jump_assisted:
        if not hasattr(env_cfg, "configure_jump_learning"):
            raise ValueError("--jump_assisted requires a Motor35 jump task")
        env_cfg.configure_jump_learning(autonomous=False)
    for name in ("min_thrust_time", "min_release_length", "flight_retract_length"):
        value = getattr(args_cli, "jump_" + name)
        if value is not None:
            if not hasattr(env_cfg, "jump") or value <= 0:
                raise ValueError("Jump overrides require a jump task and positive values")
            if env_cfg.jump.autonomous:
                raise ValueError("Trajectory overrides require --jump_assisted; autonomous actions have no reference")
            setattr(env_cfg.jump, name, value)
    for name in ("action_rate", "action_smooth"):
        value = getattr(args_cli, "jump_" + name + "_weight")
        if value is not None:
            if not hasattr(env_cfg, "jump") or not (-float("inf") < value <= 0):
                raise ValueError("Jump smoothing weights must be finite and non-positive")
            term_name = "jump_action_smooth" if name == "action_smooth" and not env_cfg.jump.autonomous else name
            getattr(env_cfg.rewards, term_name).weight = value
    if hasattr(env_cfg, "jump"):
        fields = ("autonomous", "big_jump", "target", "clearance_min_m", "airborne_confirm_s",
                  "ground_confirm_s", "launch_com_vz_min", "clearance_max_tilt") if env_cfg.jump.autonomous else (
            "autonomous", "min_thrust_time", "min_release_length", "min_release_vz", "flight_retract_length",
            "crouch_length", "thrust_length", "prelanding_start_vz", "target")
        print("[INFO] Jump profile:", {name: getattr(env_cfg.jump, name) for name in fields})
    if args_cli.jump_trace_path is not None:
        if not hasattr(env_cfg, "jump") or args_cli.jump_trace_envs < 1:
            raise ValueError("Jump tracing requires a jump task and positive trace env count")
        env_cfg.jump.trace_path = os.path.abspath(args_cli.jump_trace_path)
        env_cfg.jump.trace_envs = args_cli.jump_trace_envs

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    # on_policy_runner_class = eval(agent_cfg.runner_type)
    # runner: OnPolicyRunner | OnPolicyRunnerMlp = on_policy_runner_class(
    #     env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    # )
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # write git state to logs
    # runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        if args_cli.checkpoint_path is not None:
            resume_path = args_cli.checkpoint_path
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer,
                    load_task_state=hasattr(env_cfg, "getup") and args_cli.getup_stage is None)
        # Recreate episodes using the restored curriculum level, never stale root states.
        if hasattr(env_cfg, "getup"):
            env_cfg.getup.initial_level = env.unwrapped.task_state.level
            env.reset()

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as file:
        pickle.dump(env_cfg, file)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as file:
        pickle.dump(agent_cfg, file)
    dump_yaml(os.path.join(log_dir, "params", "continuation.yaml"), {
        "checkpoint_path": args_cli.checkpoint_path,
        "reset_optimizer": args_cli.reset_optimizer,
    })

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=(not hasattr(env_cfg, "getup")
                                        and not getattr(env_cfg, "deterministic_baseline", False)))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
