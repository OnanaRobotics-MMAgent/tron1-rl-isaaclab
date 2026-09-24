"""Recovery resets preserving the trained flat policy's joint reference pose."""

from pathlib import Path

import torch

from .events import reset_fallen
from .nominal_geometry import nominal_collision_corners
from .state import get_state


def prepare_progressive(env, env_ids):
    robot = env.scene['robot']
    leg_ids, _ = robot.find_joints('(abad|hip|knee)_[LR]_Joint')
    limits = robot.data.joint_pos_limits[:, leg_ids]
    if not torch.isfinite(limits).all() or (limits[..., 0] >= limits[..., 1]).any():
        raise ValueError('Recovery requires finite authored leg joint limits')
    # Keep the USD limits, including continuous wheels. No articulation-wide
    # limit rewrite: it would replace infinite wheel limits with hard stops.
    urdf = Path(__file__).resolve().parents[3] / 'assets/urdf/WF_TRON1A.urdf'
    positions = dict(zip(robot.joint_names, robot.data.default_joint_pos[0].cpu().tolist()))
    env._getup_collision_corners = torch.as_tensor(
        nominal_collision_corners(urdf, positions), device=env.device)
    get_state(env)


def reset_progressive(env, env_ids):
    reset_fallen(env, env_ids)
    robot = env.scene['robot']
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot.set_joint_position_target(robot.data.default_joint_pos[env_ids], env_ids=env_ids)
    robot.set_joint_velocity_target(torch.zeros_like(robot.data.default_joint_vel[env_ids]), env_ids=env_ids)
