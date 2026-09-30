"""Dense recovery rewards with posture regularization gated near standing."""

import math

import torch

from .state import get_state
from .limits import outside_joint_limits


def upright(env):
    # Unlike gravity_xy squared, the signed z component distinguishes upside-down.
    return ((1.0 - env.scene["robot"].data.projected_gravity_b[:, 2]) * 0.5).clamp(0.0, 1.0)


def height_tracking(env):
    height = env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return upright(env) * torch.exp(-((height - env.cfg.getup.target_height) / 0.25).square())


def signed_height_tracking(env):
    height = env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    up_z = -env.scene["robot"].data.projected_gravity_b[:, 2]
    orientation = ((up_z + 1.0) * 0.5).clamp(0.0, 1.0).square()
    return orientation * torch.exp(-((height - env.cfg.getup.target_height) / 0.25).square())


def height_progress(env):
    """Bounded dense height signal below the narrow final-standing kernel."""
    height = env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    low_height = max(0.05, env.cfg.getup.target_height * 0.35)
    return upright(env) * ((height - low_height) / (env.cfg.getup.target_height - low_height)).clamp(0.0, 1.0)


def standing_gate(env):
    height = env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    low_height = max(0.05, env.cfg.getup.target_height * 0.35)
    height_scale = max(0.01, env.cfg.getup.target_height - low_height)
    return ((upright(env) - 0.8) / 0.2).clamp(0.0, 1.0) * ((height - low_height) / height_scale).clamp(0.0, 1.0)


def wheel_support(env, sensor_cfg):
    forces = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return standing_gate(env) * (forces > env.cfg.getup.min_wheel_force).all(dim=1).float()


def nominal_pose(env, asset_cfg):
    data = env.scene[asset_cfg.name].data
    error = (data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids]).square().mean(dim=1)
    return standing_gate(env) * torch.exp(-error / 0.25)


def quiet_standing(env):
    data = env.scene["robot"].data
    error = data.root_lin_vel_w.square().sum(dim=1) + 0.25 * data.root_ang_vel_w.square().sum(dim=1)
    return standing_gate(env) * torch.exp(-error / 0.25)


def stable_mask(env, wheel_cfg, body_cfg, leg_cfg):
    data = env.scene["robot"].data
    cfg = env.cfg.getup
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    forces = env.scene[wheel_cfg.name].data.net_forces_w[:, wheel_cfg.body_ids, 2]
    body_forces = env.scene[body_cfg.name].data.net_forces_w[:, body_cfg.body_ids].norm(dim=-1)
    pose_error = (data.joint_pos[:, leg_cfg.joint_ids] - data.default_joint_pos[:, leg_cfg.joint_ids]).square().mean(dim=1)
    return ((-data.projected_gravity_b[:, 2] > math.cos(cfg.success_tilt))
            & ((height - cfg.target_height).abs() < cfg.height_tolerance)
            & (data.root_lin_vel_w.norm(dim=1) < cfg.max_linear_speed)
            & (data.root_ang_vel_w.norm(dim=1) < cfg.max_angular_speed)
            & (forces > cfg.min_wheel_force).all(dim=1)
            & (body_forces < cfg.max_body_force).all(dim=1)
            & (pose_error < cfg.max_pose_error))


def stable_reward(env, wheel_cfg, body_cfg, leg_cfg):
    return stable_mask(env, wheel_cfg, body_cfg, leg_cfg).float()


def success_bonus(env):
    # Terminations run before rewards; the success flag is cleared on reset.
    # RewardManager multiplies by dt, so this produces a fixed event bonus.
    return get_state(env).success.float() / env.step_dt


def excessive_impact(env, sensor_cfg, threshold=600.0):
    force = env.scene[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1)
    return ((force.amax(dim=1) - threshold).clamp(min=0.0) / threshold).square().sum(dim=1)


def wheel_speed_near_stand(env, asset_cfg):
    return standing_gate(env) * env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids].square().sum(dim=1)


def leg_limit_failure(env, asset_cfg):
    data = env.scene[asset_cfg.name].data
    failed = outside_joint_limits(data.joint_pos[:, asset_cfg.joint_ids],
                                  data.joint_pos_limits[:, asset_cfg.joint_ids],
                                  env.cfg.getup.joint_limit_tolerance)
    return failed.float() / env.step_dt


def low_height_deficit(env):
    """Penalize remaining low, with a smooth gradient and a reset grace period."""
    height = env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    low_height = max(0.05, env.cfg.getup.target_height * 0.35)
    deficit = ((env.cfg.getup.target_height - height) / (env.cfg.getup.target_height - low_height)).clamp(0., 1.)
    return deficit * (get_state(env).steps * env.step_dt > 1.0)
