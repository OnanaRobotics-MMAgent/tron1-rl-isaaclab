"""Motor35 position-controlled adaptation of Wheel-Legged-Lab clearance jumping.

Reference: zyicome/Wheel-Legged-Lab, e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7.
Phase/reward design follows its BSD-3-Clause jump modules (Copyright 2026 zyicome).
Modified for Motor35 URDF kinematics, joint-position actions and absolute wheel
bottom clearance. No virtual-model torques or phase-dependent gains are used.
"""

import math

import torch

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass


IDLE, CROUCH, THRUST, FLIGHT, LANDING, RECOVERY = range(6)
LEG_NAMES = ["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
             "knee_L_Joint", "knee_R_Joint"]
# Hip->knee and knee->wheel origins in Motor35_WF.urdf, sagittal x/z.
UPPER = (-0.0984349768784346, -0.0176225800307687)
LOWER = (0.0974505224847294, -0.0510234815300838)


@configclass
class JumpCfg:
    target: float = 0.17
    big_jump: bool = False
    wheel_radius: float = 0.04
    nominal_length: float = 0.1402007278
    nominal_angle: float = -0.1994046682
    crouch_length: float = 0.115
    crouch_ready_length: float = 0.13
    thrust_length: float = 0.200
    min_release_length: float = 0.190
    flight_retract_length: float = 0.110
    prelanding_length: float = 0.195
    landing_absorption_length: float = 0.125
    crouch_min_time: float = 0.24
    max_crouch_time: float = 0.48
    min_thrust_time: float = 0.20
    max_thrust_time: float = 0.36
    min_release_vz: float = 0.50
    min_takeoff_vz: float = 0.10
    max_airborne_wait: float = 0.15
    min_flight_time: float = 0.04
    landing_time: float = 0.30
    landing_compression_time: float = 0.20
    prelanding_start_vz: float = -0.08
    prelanding_full_vz: float = -0.65
    recovery_stable_time: float = 0.50
    max_recovery_time: float = 2.0
    contact_force_threshold: float = 2.0
    contact_confirm_steps: int = 2
    start_max_tilt: float = 0.30
    start_max_speed: float = 0.40
    recovery_max_tilt: float = 0.20
    recovery_max_vz: float = 0.30
    success_height_ratio: float = 0.75
    success_min_air_time: float = 0.12
    success_max_landing_speed: float = 1.10
    soft_landing_speed: float = 0.80
    # Same residual fractions as C2, applied to the existing 0.12 rad scale.
    leg_action_residual_scale: float = 0.05
    prelanding_action_residual_scale: float = 0.08
    landing_action_residual_scale: float = 0.15
    trigger_delay_range: tuple = (0.7, 1.1)
    resampling_time_range: tuple = (3.5, 4.5)
    trigger_pulse_time: float = 0.10
    jump_probability: float = 0.90


def leg_kinematics(q):
    """FK from ordered L/R abad, hip, knee angles; returns length and pitch.

    Both sides use the left-leg canonical pitch angles. Abduction rotates the
    whole sagittal chain, so it does not change its virtual length/pitch.
    """
    sign = q.new_tensor([1.0, -1.0])
    hip, knee = q[..., 2:4] * sign, q[..., 4:6] * sign
    distal = hip - knee
    x = hip.cos() * UPPER[0] + hip.sin() * UPPER[1]
    x += distal.cos() * LOWER[0] + distal.sin() * LOWER[1]
    z = -hip.sin() * UPPER[0] + hip.cos() * UPPER[1]
    z += -distal.sin() * LOWER[0] + distal.cos() * LOWER[1]
    return torch.sqrt(x.square() + z.square()), torch.atan2(x, -z)


def leg_pose(length, angle):
    """Analytic IK on Motor35's nominal branch, without a force controller."""
    a, b = math.hypot(*UPPER), math.hypot(*LOWER)
    u, v = math.atan2(UPPER[1], UPPER[0]), math.atan2(LOWER[1], LOWER[0])
    delta = torch.acos(((length.square() - a*a - b*b) / (2*a*b)).clamp(-1, 1))
    knee = delta - (v - u)
    x = UPPER[0] + knee.cos()*LOWER[0] - knee.sin()*LOWER[1]
    z = UPPER[1] + knee.sin()*LOWER[0] + knee.cos()*LOWER[1]
    hip = torch.atan2(z, x) - (angle - math.pi / 2)
    hip = torch.atan2(hip.sin(), hip.cos())
    zero = torch.zeros_like(length)
    return torch.stack((zero, zero, hip, -hip, knee, -knee), dim=-1)


def wheel_clearances(env, asset_cfg, radius):
    robot = env.scene[asset_cfg.name]
    return robot.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3] - radius


def jump_measurements(env, asset_cfg, sensor_cfg, radius, threshold=2.0):
    clearance = wheel_clearances(env, asset_cfg, radius)
    forces = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
    contact = forces > threshold
    airborne = (~contact).all(dim=-1) & (clearance > 0.001).all(dim=-1)
    return clearance.min(dim=-1).values, airborne, contact.all(dim=-1)


def success_mask(had_flight, landed, peak, rise, air_time, landing_vz, cfg):
    return (had_flight & landed & (peak >= cfg.target)
            & (rise >= cfg.success_height_ratio * cfg.target)
            & (air_time >= cfg.success_min_air_time)
            & (landing_vz.abs() <= cfg.success_max_landing_speed))


class JumpState:
    """Updated once after physics, before rewards; reset only for selected envs."""

    def __init__(self, env):
        self.env, self.cfg = env, env.cfg.jump
        robot = env.scene["robot"]
        self.leg_ids = robot.find_joints(LEG_NAMES, preserve_order=True)[0]
        self.wheel_ids = robot.find_bodies(["wheel_L_Link", "wheel_R_Link"], preserve_order=True)[0]
        self.sensor_ids = env.scene["contact_forces"].find_bodies(
            ["wheel_L_Link", "wheel_R_Link"], preserve_order=True)[0]
        self.phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.contact = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.bool)
        self.length = torch.zeros(env.num_envs, 2, device=env.device)
        self.angle = torch.zeros_like(self.length)
        self.length_speed = torch.zeros_like(self.length)
        self.action_history = torch.zeros(env.num_envs, 3, 6, device=env.device)
        for name in ("time", "cycle", "next_cycle", "delay", "start_z", "rise", "peak",
                     "air_time", "air_run", "air_steps", "ground_steps", "stable_time",
                     "landing_vz", "takeoff_vz", "clearance", "attempts", "takeoffs",
                     "successes", "soft_landings", "episode_peak", "episode_raw_peak", "fail_crouch", "fail_thrust",
                     "fail_unload", "fail_performance", "fail_recovery"):
            setattr(self, name, torch.zeros(env.num_envs, device=env.device))
        for name in ("enabled", "had_flight", "landed", "airborne", "takeoff_event",
                     "landing_event", "success_event", "failure_event"):
            setattr(self, name, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
        self.reset(slice(None))

    def sample_cycle(self, ids):
        n = self.cycle[ids].numel()
        self.cycle[ids] = 0
        self.next_cycle[ids] = torch.empty(n, device=self.env.device).uniform_(*self.cfg.resampling_time_range)
        self.delay[ids] = torch.empty(n, device=self.env.device).uniform_(*self.cfg.trigger_delay_range)
        self.enabled[ids] = torch.rand(n, device=self.env.device) < self.cfg.jump_probability

    def reset(self, ids):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                value[ids] = 0
        q = self.env.scene["robot"].data.joint_pos[:, self.leg_ids]
        length, angle = leg_kinematics(q)
        self.length[ids], self.angle[ids] = length[ids], angle[ids]
        self.sample_cycle(ids)

    def enter(self, mask, phase):
        self.phase[mask], self.time[mask] = phase, 0.0
        if phase in (CROUCH, THRUST, FLIGHT):
            self.air_steps[mask] = 0
        if phase in (CROUCH, FLIGHT):
            self.ground_steps[mask] = 0
        if phase in (CROUCH, LANDING, RECOVERY):
            self.stable_time[mask] = 0

    def fail(self, mask, reason, phase=RECOVERY):
        self.failure_event |= mask
        getattr(self, "fail_" + reason)[mask] += 1
        self.enter(mask, phase)

    def update(self):
        env, c, dt = self.env, self.cfg, self.env.step_dt
        robot = env.scene["robot"].data
        for name in ("takeoff_event", "landing_event", "success_event", "failure_event"):
            getattr(self, name).zero_()
        self.time += dt
        self.cycle += dt
        self.sample_cycle(self.cycle >= self.next_cycle)
        length, self.angle = leg_kinematics(robot.joint_pos[:, self.leg_ids])
        self.length_speed = (length - self.length) / dt
        self.length[:] = length
        self.action_history[:, :2] = self.action_history[:, 1:].clone()
        self.action_history[:, 2] = env.action_manager.get_term("joint_pos").raw_actions
        forces = env.scene["contact_forces"].data.net_forces_w[:, self.sensor_ids]
        self.contact[:] = forces.norm(dim=-1) > c.contact_force_threshold
        self.clearance[:] = (robot.body_pos_w[:, self.wheel_ids, 2]
                             - env.scene.env_origins[:, 2:3] - c.wheel_radius).amin(dim=1)
        self.airborne[:] = ~self.contact.any(dim=1) & (self.clearance > 0.001)
        self.air_steps = torch.where(self.airborne, self.air_steps + 1, 0)
        self.ground_steps = torch.where(self.contact.any(dim=1), self.ground_steps + 1, 0)
        z, vz = robot.root_pos_w[:, 2], robot.root_lin_vel_w[:, 2]
        tilt = torch.acos((-robot.projected_gravity_b[:, 2]).clamp(-1, 1))
        active = self.phase != IDLE
        self.rise = torch.where(active, torch.maximum(self.rise, z - self.start_z), self.rise)
        confirmed = (self.phase == FLIGHT) & self.airborne & self.had_flight
        self.peak = torch.where(confirmed, torch.maximum(self.peak, self.clearance), self.peak)
        self.episode_peak = torch.maximum(self.episode_peak, self.peak)
        self.episode_raw_peak = torch.where(active & self.airborne,
            torch.maximum(self.episode_raw_peak, self.clearance), self.episode_raw_peak)
        self.air_run = torch.where(confirmed, self.air_run + dt, 0)
        self.air_time = torch.maximum(self.air_time, self.air_run)
        start = ((self.phase == IDLE) & self.enabled & (self.cycle >= self.delay)
                 & (self.cycle < self.delay + c.trigger_pulse_time) & self.contact.all(dim=1)
                 & (tilt < c.start_max_tilt)
                 & (robot.root_lin_vel_b[:, :2].norm(dim=1) < c.start_max_speed))
        self.start_z[start] = z[start]
        for name in ("rise", "peak", "air_time", "air_run", "landing_vz", "takeoff_vz"):
            getattr(self, name)[start] = 0
        self.had_flight[start], self.landed[start] = False, False
        self.attempts[start] += 1
        self.enter(start, CROUCH)
        mean_length = self.length.mean(dim=1)
        self.enter((self.phase == CROUCH) & (self.time >= c.crouch_min_time)
                   & (mean_length <= c.crouch_ready_length), THRUST)
        self.fail((self.phase == CROUCH) & (self.time >= c.max_crouch_time), "crouch")
        self.enter((self.phase == THRUST) & (self.time >= c.min_thrust_time)
                   & (vz >= c.min_release_vz) & (mean_length >= c.min_release_length), FLIGHT)
        takeoff = ((self.phase == FLIGHT) & ~self.had_flight
                   & (self.air_steps >= c.contact_confirm_steps) & (vz >= c.min_takeoff_vz))
        self.takeoff_event[:] = takeoff
        self.had_flight |= takeoff
        self.takeoff_vz[takeoff] = vz[takeoff]
        self.takeoffs[takeoff] += 1
        self.fail((self.phase == THRUST) & (self.time >= c.max_thrust_time), "thrust")
        self.fail((self.phase == FLIGHT) & ~self.had_flight
                  & (self.time >= c.max_airborne_wait), "unload")
        first = ((self.phase == FLIGHT) & self.had_flight & ~self.landed
                 & (self.ground_steps == 1) & (vz <= 0))
        self.landing_vz[first] = vz[first]
        self.landed |= first
        landing = ((self.phase == FLIGHT) & self.had_flight & self.landed
                   & (self.time >= c.min_flight_time) & (self.ground_steps >= c.contact_confirm_steps)
                   & (vz <= 0))
        self.landing_event[:] = landing
        self.soft_landings += (landing & (self.landing_vz.abs() <= c.soft_landing_speed)).float()
        self.enter(landing, LANDING)
        self.enter((self.phase == LANDING) & (self.time >= c.landing_time), RECOVERY)
        stable = ((self.phase == RECOVERY) & self.contact.all(dim=1)
                  & (tilt < c.recovery_max_tilt) & (vz.abs() < c.recovery_max_vz))
        self.stable_time = torch.where(stable, self.stable_time + dt, 0)
        recovered = (self.phase == RECOVERY) & (self.stable_time >= c.recovery_stable_time)
        success = recovered & success_mask(self.had_flight, self.landed, self.peak,
                                           self.rise, self.air_time, self.landing_vz, c)
        self.success_event[:] = success
        self.successes += success.float()
        self.fail(recovered & self.had_flight & ~success, "performance")
        self.enter(recovered, IDLE)
        self.fail((self.phase == RECOVERY) & (self.time >= c.max_recovery_time), "recovery", IDLE)

    def reference(self):
        c = self.cfg
        target = torch.full_like(self.time, c.nominal_length)
        target = torch.where(self.phase == CROUCH, c.crouch_length, target)
        target = torch.where(self.phase == THRUST, c.thrust_length, target)
        vz = self.env.scene["robot"].data.root_lin_vel_w[:, 2]
        descent = ((c.prelanding_start_vz - vz) /
                   (c.prelanding_start_vz - c.prelanding_full_vz)).clamp(0, 1)
        flight = c.flight_retract_length + descent * (c.prelanding_length - c.flight_retract_length)
        target = torch.where(self.phase == FLIGHT, flight, target)
        compression = (self.time / c.landing_compression_time).clamp(0, 1)
        landing = c.prelanding_length + compression * (c.landing_absorption_length - c.prelanding_length)
        return torch.where(self.phase == LANDING, landing, target)

    def metrics(self, ids):
        attempts = self.attempts[ids].sum().clamp_min(1)
        result = {"jump/" + name: getattr(self, name)[ids].sum() / attempts for name in
                  ("takeoffs", "successes", "soft_landings", "fail_crouch", "fail_thrust",
                   "fail_unload", "fail_performance", "fail_recovery")}
        result["jump/peak_clearance_m"] = self.episode_peak[ids].mean()
        result["jump/max_clearance_m"] = self.episode_peak[ids].max()
        result["jump/raw_max_clearance_m"] = self.episode_raw_peak[ids].max()
        result["jump/attempts_per_episode"] = self.attempts[ids].mean()
        return result


def get_state(env):
    if not hasattr(env, "_motor35_jump"):
        env._motor35_jump = JumpState(env)
    return env._motor35_jump


class JumpUpdate(ManagerTermBase):
    """Non-terminating first termination term: run after physics, before rewards."""

    def __call__(self, env):
        state = get_state(env)
        state.update()
        return torch.zeros_like(state.had_flight)

    def reset(self, env_ids=None):
        state = get_state(self._env)
        ids = slice(None) if env_ids is None else env_ids
        metrics = state.metrics(ids)
        # TerminationManager discards term reset return values in IsaacLab 2.x.
        self._env.extras.setdefault("log", {}).update(metrics)
        state.reset(ids)
        return metrics


class JumpLegPositionAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        s = get_state(self._env)
        active = (s.phase != IDLE) & (s.phase != RECOVERY)
        scale = torch.full_like(s.time, s.cfg.leg_action_residual_scale)
        scale = torch.where(s.phase == FLIGHT, s.cfg.prelanding_action_residual_scale, scale)
        scale = torch.where(s.phase == LANDING, s.cfg.landing_action_residual_scale, scale)
        default = self._asset.data.default_joint_pos[:, self._joint_ids]
        reference = leg_pose(s.reference(), s.cfg.nominal_angle)
        assisted = reference + (self._processed_actions - default) * scale.unsqueeze(-1)
        self._processed_actions[:] = torch.where(active.unsqueeze(-1), assisted, self._processed_actions)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._processed_actions.clamp_(min=limits[..., 0], max=limits[..., 1])


def jump_time_obs(env):
    # Keep the three-column checkpoint expansion; phase replaces the old clock.
    return get_state(env).phase.float().unsqueeze(-1) / 5.0


def jump_clearance_obs(env, asset_cfg, radius):
    return wheel_clearances(env, asset_cfg, radius).amin(dim=-1).unsqueeze(-1)


def jump_vertical_speed_obs(env, asset_cfg):
    return env.scene[asset_cfg.name].data.root_lin_vel_w[:, 2:3]


def jump_reward(env, kind):
    """C2 reward formulas with Motor35 geometry and zero motion commands."""
    s, r = get_state(env), env.scene["robot"].data
    c = s.cfg
    gate = lambda *phases: sum((s.phase == p).float() for p in phases)
    gaussian = lambda error, std: torch.exp(-error.square() / std**2)
    tilt = torch.acos((-r.projected_gravity_b[:, 2]).clamp(-1, 1))
    upright = (-r.projected_gravity_b[:, 2]).clamp(0, .7) / .7
    vz = r.root_lin_vel_w[:, 2]
    active = gate(CROUCH, THRUST, FLIGHT, LANDING)
    if kind == "lin_vel_z":
        return r.root_lin_vel_b[:, 2].square() * gate(IDLE, LANDING, RECOVERY)
    if kind == "base_height":
        # Grounded nominal root height: hip offset + sagittal leg z + radius.
        target = .02625 + c.nominal_length * math.cos(c.nominal_angle) + c.wheel_radius
        height = r.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        return gaussian(height - target, .04) * upright * gate(IDLE, RECOVERY)
    if kind == "action_smooth":
        return (s.action_history[:, 2] - 2*s.action_history[:, 1] + s.action_history[:, 0]).square().sum(-1)
    if kind == "nominal_state":
        return (s.angle[:, 0] - s.angle[:, 1]).square() + .25*((s.length[:, 0]-s.length[:, 1])/.067).square()
    if kind == "track_lin_vel":
        return gaussian(r.root_lin_vel_b[:, 0], .25) * upright
    if kind == "track_ang_vel":
        return gaussian(r.root_ang_vel_b[:, 2], .5) * upright
    if kind == "track_heading":
        w, x, y, z = r.root_quat_w.unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        return gaussian(yaw, .5) * upright
    if kind in ("crouch", "thrust_pose", "landing_pose"):
        target = c.thrust_length if kind == "thrust_pose" else s.reference()
        std = {"crouch": .017, "thrust_pose": .020, "landing_pose": .015}[kind]
        mask = {"crouch": gate(CROUCH), "thrust_pose": gate(THRUST), "landing_pose": gate(FLIGHT, LANDING)}[kind]
        return gaussian(s.length.mean(-1) - target, std) * mask
    if kind == "phase_action":
        target_q = env.action_manager.get_term("joint_pos").processed_actions
        target_length, _ = leg_kinematics(target_q)
        error = ((target_length - s.reference().unsqueeze(-1)) / .06).square().mean(-1)
        return (1-.75*error).clamp_min(0) * active
    if kind == "thrust_speed":
        return (s.length_speed.mean(-1) / .8).clamp(0, 1) * gate(THRUST)
    if kind == "takeoff":
        target_vz = math.sqrt(2*9.81*c.target)
        return (.7*(vz/target_vz).clamp(0, 1)+.3*gaussian(vz-target_vz, .60))*gate(THRUST)
    if kind in ("height", "wheel_clearance"):
        height = r.root_pos_w[:, 2] - s.start_z if kind == "height" else s.clearance
        mask = gate(FLIGHT) * (s.had_flight & s.airborne).float()
        if c.big_jump:
            return height.clamp_min(0)/c.target * mask
        fraction, std = (.6, .04) if kind == "height" else (.7, .025)
        return (fraction*(height/c.target).clamp(0, 1)+(1-fraction)*gaussian(height-c.target, std))*mask
    if kind == "airborne":
        return (s.airborne & s.had_flight).float()*gate(FLIGHT)
    if kind == "symmetry":
        return torch.exp(-((s.length[:, 0]-s.length[:, 1])/.010).square()
                         -((s.angle[:, 0]-s.angle[:, 1])/.10).square())*active
    if kind == "landing_soft":
        return gaussian(s.landing_vz, .60)*gaussian(tilt, .18)*s.landing_event.float()
    if kind == "landing_impact":
        return s.landing_vz.square()*s.landing_event.float()
    if kind == "recovery":
        return gaussian(vz, .25)*gaussian(tilt, .18)*s.contact.all(-1)*s.had_flight*gate(RECOVERY)
    if kind in ("takeoff_event", "success", "failure"):
        return getattr(s, kind if kind.endswith("_event") else kind+"_event").float()
    raise ValueError(f"Unknown jump reward: {kind}")


def jump_minimum_height(env, minimum_height=.05):
    return env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] < minimum_height


def jump_joint_margin(env, asset_cfg, margin=.15):
    robot = env.scene[asset_cfg.name].data
    pos = robot.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    return ((margin - (pos - limits[..., 0])).clamp_min(0)
            + (margin - (limits[..., 1] - pos)).clamp_min(0)).sum(-1)


def jump_leg_posture(env, asset_cfg):
    robot = env.scene[asset_cfg.name].data
    return (robot.joint_pos[:, asset_cfg.joint_ids]
            - robot.default_joint_pos[:, asset_cfg.joint_ids]).square().sum(-1)
