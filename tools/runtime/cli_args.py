from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument("--getup_stage", type=int, default=None,
                           help="Initial GetUp level; valid range depends on the selected task.")
    arg_group.add_argument("--getup_fixed_stage", action="store_true",
                           help="Disable automatic GetUp curriculum and easier-pose replay.")
    arg_group.add_argument("--getup_tilt_range", type=float, nargs=2, default=None, metavar=("MIN_DEG", "MAX_DEG"),
                           help="Override the selected stage's tilt range in degrees (e.g. 15 25).")
    arg_group.add_argument("--pose_bank", type=str, default=None,
                           help="Settled random-joint pose bank for the Fallen recovery task.")


def configure_getup(env_cfg, args_cli):
    """Apply explicit task overrides before creating the simulation."""
    if args_cli.pose_bank is not None:
        if not hasattr(env_cfg, "fallen"):
            raise ValueError("--pose_bank requires a Fallen recovery task")
        env_cfg.fallen.bank_path = args_cli.pose_bank
    if not hasattr(env_cfg, "getup"):
        if args_cli.getup_stage is not None or args_cli.getup_fixed_stage or args_cli.getup_tilt_range is not None:
            raise ValueError("--getup_stage/--getup_fixed_stage require a GetUp task.")
        return
    if args_cli.getup_stage is not None:
        if not 0 <= args_cli.getup_stage < len(env_cfg.getup.tilt_ranges_deg):
            raise ValueError("GetUp stage is outside this task curriculum.")
        env_cfg.getup.initial_level = args_cli.getup_stage
    if args_cli.getup_fixed_stage:
        env_cfg.getup.curriculum_enabled = False
    if args_cli.getup_tilt_range is not None:
        low, high = args_cli.getup_tilt_range
        if not 0 <= low < high <= 180:
            raise ValueError("GetUp tilt range must satisfy 0 <= min < max <= 180 degrees.")
        ranges = list(env_cfg.getup.tilt_ranges_deg)
        ranges[env_cfg.getup.initial_level] = (low, high)
        env_cfg.getup.tilt_ranges_deg = tuple(ranges)


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
