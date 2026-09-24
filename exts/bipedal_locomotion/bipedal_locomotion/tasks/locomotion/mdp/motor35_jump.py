"""Motor35 in-place jump control and wheel-clearance metrics."""

import torch

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import ManagerTermBase


def jump_phase(env):
    """A time-based reference, shared by action processing and observations."""
    return env.episode_length_buf.float() * env.step_dt


def reference_pose(time, default_pose):
    """Mirrored hip/knee targets; the policy supplies the residual."""
    crouch = default_pose.new_tensor([0.0, 0.0, -0.25, 0.25, -0.55, 0.55])
    extend = default_pose.new_tensor([0.0, 0.0, -1.10, 1.10, -2.15, 2.15])
    def blend(start, end, src, dst):
        alpha = ((time - start) / (end - start)).clamp(0.0, 1.0).unsqueeze(-1)
        return src + alpha * (dst - src)

    crouching = blend(0.10, 0.30, default_pose, crouch)
    pushing = blend(0.50, 0.72, crouch, extend)
    recovering = blend(0.72, 1.25, extend, default_pose)
    return torch.where((time < 0.50).unsqueeze(-1), crouching,
                       torch.where((time < 0.72).unsqueeze(-1), pushing, recovering))


class JumpLegPositionAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        default_pose = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._processed_actions += reference_pose(jump_phase(self._env), default_pose) - default_pose
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._processed_actions.clamp_(min=limits[..., 0], max=limits[..., 1])


def wheel_clearances(env, asset_cfg, radius):
    robot = env.scene[asset_cfg.name]
    return robot.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3] - radius


def jump_measurements(env, asset_cfg, sensor_cfg, radius, threshold=1.0):
    clearance = wheel_clearances(env, asset_cfg, radius)
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
    contact = forces > threshold
    # Both wheels must have separated: a tilted one-wheel lift is not a jump.
    airborne = (~contact).all(dim=-1) & (clearance > 0.01).all(dim=-1)
    return clearance.min(dim=-1).values, airborne, contact.all(dim=-1)


def jump_time_obs(env):
    return (jump_phase(env) / 2.0).clamp(0.0, 1.0).unsqueeze(-1)


def jump_clearance_obs(env, asset_cfg, radius):
    return wheel_clearances(env, asset_cfg, radius).min(dim=-1).values.unsqueeze(-1)


def jump_vertical_speed_obs(env, asset_cfg):
    return env.scene[asset_cfg.name].data.root_lin_vel_w[:, 2:3]


def jump_height_step(previous_peak, clearance, airborne, target, big_jump):
    current = torch.where(airborne, clearance.clamp_min(0.0), previous_peak)
    peak = torch.maximum(previous_peak, current)
    increment = peak - previous_peak
    if big_jump:
        reward = increment / target
    else:
        reward = (peak.clamp(max=target) - previous_peak.clamp(max=target)) / target
    return peak, reward


def landing_success(had_flight, was_airborne, both_contact, peak, target, tilt, vertical_speed):
    return (had_flight & was_airborne & both_contact & (peak >= target)
            & (tilt < 0.35) & (vertical_speed.abs() < 1.0))


class JumpOutcome(ManagerTermBase):
    """One stateful term to account for a full takeoff/flight/landing cycle."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.peak = torch.zeros(env.num_envs, device=env.device)
        self.had_flight = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.was_airborne = torch.zeros_like(self.had_flight)
        self.touched_down = torch.zeros_like(self.had_flight)
        self.rewarded_landing = torch.zeros_like(self.had_flight)

    def reset(self, env_ids=None):
        self.peak[env_ids] = 0.0
        self.had_flight[env_ids] = False
        self.was_airborne[env_ids] = False
        self.touched_down[env_ids] = False
        self.rewarded_landing[env_ids] = False

    def __call__(self, env, asset_cfg, sensor_cfg, radius, target, big_jump=False):
        height, airborne, both_contact = jump_measurements(env, asset_cfg, sensor_cfg, radius)
        self.peak[:], progress = jump_height_step(self.peak, height, airborne, target, big_jump)
        robot = env.scene[asset_cfg.name]
        tilt = robot.data.projected_gravity_b[:, :2].norm(dim=-1)
        vertical_speed = robot.data.root_lin_vel_w[:, 2]
        self.touched_down |= self.had_flight & self.was_airborne & both_contact
        success = landing_success(self.had_flight, self.touched_down, both_contact,
                                  self.peak, target, tilt, vertical_speed)
        fresh = success & ~self.rewarded_landing
        self.rewarded_landing |= fresh
        self.had_flight |= airborne
        self.was_airborne[:] = airborne
        return progress + fresh.float() * 2.0


def jump_stationary(env, asset_cfg):
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_w
    return velocity[:, :2].square().sum(dim=-1)


def jump_wheel_speed(env, asset_cfg):
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids].square().sum(dim=-1)


def jump_reference_error(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    pose = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return (pose - reference_pose(jump_phase(env), default)).square().sum(dim=-1)


def jump_takeoff_velocity(env, asset_cfg, sensor_cfg, radius):
    _, _, grounded = jump_measurements(env, asset_cfg, sensor_cfg, radius)
    time = jump_phase(env)
    robot = env.scene[asset_cfg.name]
    upright = robot.data.projected_gravity_b[:, :2].norm(dim=-1) < 0.35
    return (robot.data.root_lin_vel_w[:, 2].clamp(0.0, 2.0)
            * (grounded & upright & (time >= 0.5) & (time < 0.85)).float())
