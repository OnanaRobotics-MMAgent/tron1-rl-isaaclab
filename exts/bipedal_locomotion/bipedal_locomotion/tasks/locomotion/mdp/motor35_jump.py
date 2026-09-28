"""Autonomous Motor35 clearance learning and historical assisted comparisons.

Reference: zyicome/Wheel-Legged-Lab, e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7.
The opt-in assisted phase/reward design follows its BSD-3-Clause jump modules
(Copyright 2026 zyicome). Autonomous mode has no reference-action controller.
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
    autonomous: bool = False
    clearance_min_m: float = 0.01
    airborne_confirm_s: float = 0.04
    ground_confirm_s: float = 0.04
    launch_com_vz_min: float = 0.05
    clearance_max_tilt: float = 0.50
    trace_path: str | None = None
    trace_envs: int = 4
    capture_diagnostics: bool = False
    drive_diagnostics: bool = False
    thrust_ramp_time: float = 0.0
    residual_limit_rad: float | None = None
    flight_leg_angle: float | None = None
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
        if self.cfg.trace_path:
            import csv
            from pathlib import Path
            path = Path(self.cfg.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            header = [
                    "step", "env", "episode_step", "phase_before", "phase_time_s",
                    "vz_mps", "length_left_m", "length_right_m", "reference_m",
                    "clearance_m", "contact_left_n", "contact_right_n", "tilt_rad",
                    "leg_torque_max_nm", "time_gate", "velocity_gate", "length_gate",
                    "phase_after", "takeoff_event", "failure_event"]
            if self.cfg.autonomous:
                header[4], header[5], header[8] = "flight_elapsed_s", "com_vz_mps", "unused_reference_m"
                header[14:17] = ["air_time_gate", "upward_com_gate", "clearance_gate"]
            with path.open("x", newline="") as stream:
                csv.writer(stream).writerow(header)

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
        phase_before = self.phase.clone() if c.capture_diagnostics else None
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
        trace = None
        if c.trace_path:
            n = min(c.trace_envs, env.num_envs)
            fields = [env.episode_length_buf, self.phase, self.time, vz,
                      self.length[:, 0], self.length[:, 1], self.reference(), self.clearance,
                      forces[:, 0].norm(dim=-1), forces[:, 1].norm(dim=-1), tilt,
                      robot.applied_torque[:, self.leg_ids].abs().amax(dim=-1),
                      self.time >= c.min_thrust_time, vz >= c.min_release_vz,
                      self.length.mean(dim=1) >= c.min_release_length]
            trace = torch.stack([v[:n].float() for v in fields], dim=1)
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
        # Wheel touchdown can occur while the root still rises (leg extension
        # or impact rebound). Contact, not root-velocity sign, ends flight.
        first = ((self.phase == FLIGHT) & self.had_flight & ~self.landed
                 & (self.ground_steps == 1))
        self.landing_vz[first] = vz[first]
        self.landed |= first
        landing = ((self.phase == FLIGHT) & self.had_flight & self.landed
                   & (self.time >= c.min_flight_time) & (self.ground_steps >= c.contact_confirm_steps))
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
        if c.capture_diagnostics:
            # Evaluation must see the final physics frame even when the
            # environment auto-resets before returning from step().
            env._jump_step_diagnostics = {
                "start": start.clone(), "phase_before": phase_before,
                "phase_after": self.phase.clone(), "vz": vz.clone(),
                "root_z": z.clone(), "clearance": self.clearance.clone(),
                "airborne": self.airborne.clone(), "contact": self.contact.clone(),
                "takeoff": self.takeoff_event.clone(), "landing": self.landing_event.clone(),
                "success": self.success_event.clone(), "failure": self.failure_event.clone(),
                "stable_recovered": recovered.clone(),
                "landing_vz": self.landing_vz.clone(), "tilt": tilt.clone(),
                "leg_torque": robot.applied_torque[:, self.leg_ids].abs().amax(dim=1).clone(),
            }
        if trace is not None:
            import csv
            n = trace.shape[0]
            end = torch.stack([self.phase[:n], self.takeoff_event[:n], self.failure_event[:n]], dim=1)
            rows = torch.cat((trace, end.float()), dim=1).detach().cpu().tolist()
            with open(c.trace_path, "a", newline="") as stream:
                csv.writer(stream).writerows(
                    [env.common_step_counter, i, *row] for i, row in enumerate(rows))

    def reference(self):
        c = self.cfg
        target = torch.full_like(self.time, c.nominal_length)
        target = torch.where(self.phase == CROUCH, c.crouch_length, target)
        thrust = torch.full_like(target, c.thrust_length)
        if c.thrust_ramp_time > 0:
            thrust = c.crouch_length + (self.time / c.thrust_ramp_time).clamp(0, 1) * (
                c.thrust_length - c.crouch_length)
        target = torch.where(self.phase == THRUST, thrust, target)
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

    def reference_angle(self):
        c = self.cfg
        angle = torch.full_like(self.time, c.nominal_angle)
        if c.flight_leg_angle is not None:
            vz = self.env.scene["robot"].data.root_lin_vel_w[:, 2]
            descent = ((c.prelanding_start_vz - vz) /
                       (c.prelanding_start_vz - c.prelanding_full_vz)).clamp(0, 1)
            flight = c.flight_leg_angle + descent * (c.nominal_angle - c.flight_leg_angle)
            angle = torch.where(self.phase == FLIGHT, flight, angle)
        return angle


def autonomous_clearance_progress(previous_peak, current_peak, valid, target, big_jump):
    """One increment per new flight peak; no airtime or takeoff-count bonus."""
    peak = torch.where(valid, torch.maximum(previous_peak, current_peak.clamp_min(0)), previous_peak)
    before, after = previous_peak, peak
    if not big_jump:
        before, after = before.clamp(max=target), after.clamp(max=target)
    return peak, (after - before) / target


class AutonomousJumpState(JumpState):
    """Passive flight detection only: never supplies a control trajectory."""

    def __init__(self, env):
        # The legacy constructor creates bookkeeping, but no timers are sampled.
        super().__init__(env)
        for name in ("armed", "candidate", "pending"):
            setattr(self, name, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
        for name in ("ground_time", "max_com_vz", "rewarded_peak", "progress"):
            setattr(self, name, torch.zeros(env.num_envs, device=env.device))

    def sample_cycle(self, ids):
        self.enabled[ids] = True

    def update(self):
        env, c, dt = self.env, self.cfg, self.env.step_dt
        r = env.scene["robot"].data
        before = self.phase.clone()
        for name in ("takeoff_event", "landing_event", "success_event", "failure_event", "progress"):
            getattr(self, name).zero_()
        self.time += dt
        self.action_history[:, :2] = self.action_history[:, 1:].clone()
        self.action_history[:, 2] = env.action_manager.get_term("joint_pos").raw_actions
        length, self.angle = leg_kinematics(r.joint_pos[:, self.leg_ids])
        self.length_speed = (length - self.length) / dt
        self.length[:] = length
        forces = env.scene["contact_forces"].data.net_forces_w.norm(dim=-1)
        wheel_forces = forces[:, self.sensor_ids]
        self.contact[:] = wheel_forces > c.contact_force_threshold
        other = forces.clone()
        other[:, self.sensor_ids] = 0
        unsupported = (other <= 1.0).all(dim=1)
        self.clearance[:] = (r.body_pos_w[:, self.wheel_ids, 2]
            - env.scene.env_origins[:, 2:3] - c.wheel_radius).amin(dim=1)
        self.airborne[:] = ~self.contact.any(dim=1) & unsupported & (self.clearance > .001)
        mass = r.default_mass.to(env.device)
        com_velocity = (mass.unsqueeze(-1) * r.body_com_lin_vel_w).sum(1) / mass.sum(-1, keepdim=True)
        com_vz = com_velocity[:, 2]
        tilt = torch.acos((-r.projected_gravity_b[:, 2]).clamp(-1, 1))
        self.ground_time = torch.where(self.contact.all(-1), self.ground_time + dt, 0.)
        self.armed |= self.ground_time >= c.ground_confirm_s
        start = self.armed & self.airborne & ~self.candidate
        # Starting another jump is allowed even before stable-landing acceptance.
        abandoned = start & self.pending
        self.failure_event |= abandoned
        self.fail_performance += abandoned.float()
        self.pending[start] = False
        self.candidate[start] = True
        self.armed[start] = False
        self.had_flight[start], self.landed[start] = False, False
        for name in ("peak", "rewarded_peak", "air_run", "air_time", "max_com_vz", "ground_steps",
                     "stable_time", "landing_vz", "takeoff_vz", "rise", "time"):
            getattr(self, name)[start] = 0
        self.start_z[start] = r.root_pos_w[start, 2]
        self.attempts += start.float()
        self.air_run = torch.where(self.candidate & self.airborne, self.air_run + dt, 0.)
        self.air_time = torch.maximum(self.air_time, self.air_run)
        self.max_com_vz = torch.where(self.candidate & self.airborne,
            torch.maximum(self.max_com_vz, com_vz), self.max_com_vz)
        quality = (self.candidate & self.airborne & (tilt <= c.clearance_max_tilt)
                   & (com_velocity[:, :2].norm(dim=-1) <= c.start_max_speed))
        self.peak = torch.where(quality, torch.maximum(self.peak, self.clearance), self.peak)
        self.episode_raw_peak = torch.where(self.candidate & self.airborne,
            torch.maximum(self.episode_raw_peak, self.clearance), self.episode_raw_peak)
        valid = (quality & (self.air_run >= c.airborne_confirm_s)
                 & (self.max_com_vz >= c.launch_com_vz_min) & (self.peak >= c.clearance_min_m))
        self.takeoff_event[:] = valid & ~self.had_flight
        self.takeoff_vz[self.takeoff_event] = com_vz[self.takeoff_event]
        self.takeoffs += self.takeoff_event.float()
        self.had_flight |= valid
        self.rewarded_peak[:], self.progress[:] = autonomous_clearance_progress(
            self.rewarded_peak, self.peak, valid, c.target, c.big_jump)
        self.episode_peak = torch.where(valid, torch.maximum(self.episode_peak, self.peak), self.episode_peak)
        self.ground_steps = torch.where(self.candidate & self.contact.any(-1), self.ground_steps + 1, 0.)
        first = self.candidate & self.had_flight & ~self.landed & (self.ground_steps == 1)
        self.landing_vz[first] = com_vz[first]
        self.landed |= first
        touchdown = self.candidate & (self.ground_steps >= c.contact_confirm_steps)
        self.landing_event[:] = touchdown & self.had_flight
        self.soft_landings += (self.landing_event & (self.landing_vz.abs() <= c.soft_landing_speed)).float()
        missed = touchdown & ~self.had_flight
        self.fail_unload += missed.float()
        self.failure_event |= missed
        self.candidate[touchdown] = False
        self.pending |= self.landing_event
        self.time[self.landing_event] = 0
        stable = (self.pending & self.contact.all(-1) & (tilt < c.recovery_max_tilt)
                  & (com_vz.abs() < c.recovery_max_vz))
        self.stable_time = torch.where(stable, self.stable_time + dt, 0.)
        recovered = self.pending & (self.stable_time >= c.recovery_stable_time)
        self.success_event[:] = (recovered & (self.peak >= c.target)
            & (self.air_time >= c.success_min_air_time)
            & (self.landing_vz.abs() <= c.success_max_landing_speed))
        self.successes += self.success_event.float()
        failed = recovered & ~self.success_event
        self.fail_performance += failed.float()
        self.failure_event |= failed
        expired = self.pending & ~recovered & (self.time >= c.max_recovery_time)
        self.fail_recovery += expired.float()
        self.failure_event |= expired
        self.pending[recovered | expired] = False
        self.phase[:] = IDLE
        self.phase[self.candidate & self.had_flight] = FLIGHT
        self.phase[self.pending] = RECOVERY
        if c.capture_diagnostics:
            env._jump_step_diagnostics = {
                "start": start.clone(), "phase_before": before, "phase_after": self.phase.clone(),
                "vz": r.root_lin_vel_w[:, 2].clone(), "root_z": r.root_pos_w[:, 2].clone(),
                "clearance": self.clearance.clone(), "airborne": self.airborne.clone(),
                "contact": self.contact.clone(), "takeoff": self.takeoff_event.clone(),
                "landing": self.landing_event.clone(), "success": self.success_event.clone(),
                "failure": self.failure_event.clone(), "stable_recovered": recovered.clone(),
                "landing_vz": self.landing_vz.clone(), "tilt": tilt.clone(),
                "leg_torque": r.applied_torque[:, self.leg_ids].abs().amax(-1).clone(),
            }
        if c.trace_path:
            import csv
            # Autonomous column names distinguish passive gates from release commands.
            fields = [env.episode_length_buf, before, self.time, com_vz,
                self.length[:, 0], self.length[:, 1], torch.zeros_like(self.time), self.clearance,
                wheel_forces[:, 0], wheel_forces[:, 1], tilt,
                r.applied_torque[:, self.leg_ids].abs().amax(-1),
                self.air_run >= c.airborne_confirm_s, self.max_com_vz >= c.launch_com_vz_min,
                self.peak >= c.clearance_min_m, self.phase, self.takeoff_event, self.failure_event]
            rows = torch.stack([v[:c.trace_envs].float() for v in fields], dim=1).cpu().tolist()
            with open(c.trace_path, "a", newline="") as stream:
                csv.writer(stream).writerows([env.common_step_counter, i, *row] for i, row in enumerate(rows))


def get_state(env):
    if not hasattr(env, "_motor35_jump"):
        state_type = AutonomousJumpState if env.cfg.jump.autonomous else JumpState
        env._motor35_jump = state_type(env)
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


def position_pd_demand(target, position, velocity_target, velocity, stiffness, damping, effort):
    """Implicit PD demand estimate, not a measured PhysX drive torque."""
    return stiffness * (target - position) + damping * (velocity_target - velocity) + effort


def bounded_reference_action(reference, residual, limit, soft_limits):
    """Bound policy authority without permitting targets outside soft limits."""
    target = reference + residual.clamp(-limit, limit)
    return target.clamp(min=soft_limits[..., 0], max=soft_limits[..., 1])


class JumpLegPositionAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        s = get_state(self._env)
        active = (s.phase != IDLE) & (s.phase != RECOVERY)
        scale = torch.full_like(s.time, s.cfg.leg_action_residual_scale)
        scale = torch.where(s.phase == FLIGHT, s.cfg.prelanding_action_residual_scale, scale)
        scale = torch.where(s.phase == LANDING, s.cfg.landing_action_residual_scale, scale)
        default = self._asset.data.default_joint_pos[:, self._joint_ids]
        reference = leg_pose(s.reference(), s.reference_angle())
        residual = (self._processed_actions - default) * scale.unsqueeze(-1)
        assisted = reference + residual
        if s.cfg.residual_limit_rad is not None:
            assisted = bounded_reference_action(reference, residual, s.cfg.residual_limit_rad,
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids])
        self._processed_actions[:] = torch.where(active.unsqueeze(-1), assisted, self._processed_actions)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._processed_actions.clamp_(min=limits[..., 0], max=limits[..., 1])

    def apply_actions(self):
        super().apply_actions()
        s = get_state(self._env)
        if not s.cfg.drive_diagnostics:
            return
        import csv
        from pathlib import Path

        env, data, ids = self._env, self._asset.data, self._joint_ids
        pos, vel = data.joint_pos[:, ids], data.joint_vel[:, ids]
        demand = position_pd_demand(self._processed_actions, pos, data.joint_vel_target[:, ids], vel,
                                   data.joint_stiffness[:, ids], data.joint_damping[:, ids],
                                   data.joint_effort_target[:, ids])
        limits = data.joint_effort_limits[:, ids]
        estimate = demand.clamp(min=-limits, max=limits)
        path = Path(s.cfg.trace_path).with_name("drive_trace.csv")
        if not hasattr(env, "_jump_drive_stats"):
            env._jump_drive_stats = {
                "samples": torch.zeros(6, device=env.device),
                "saturated": torch.zeros(6, 6, device=env.device),
                "max_demand_nm": torch.zeros(6, 6, device=env.device),
                "error_abs_sum_rad": torch.zeros(6, 6, device=env.device),
            }
            fields = ["q_rad", "target_rad", "qd_rad_s", "demand_nm", "clipped_estimate_nm"]
            with path.open("x", newline="") as stream:
                csv.writer(stream).writerow(["control_step", "physics_step", "env", "phase"]
                    + [f"{joint}_{field}" for field in fields for joint in LEG_NAMES])
        stats = env._jump_drive_stats
        for phase in range(6):
            mask = (s.phase == phase).unsqueeze(-1)
            stats["samples"][phase] += mask.sum()
            stats["saturated"][phase] += ((demand.abs() >= .99 * limits) & mask).sum(0)
            stats["max_demand_nm"][phase] = torch.maximum(stats["max_demand_nm"][phase],
                torch.where(mask, demand.abs(), 0.).amax(0))
            stats["error_abs_sum_rad"][phase] += (mask * (self._processed_actions - pos).abs()).sum(0)
        n = min(s.cfg.trace_envs, env.num_envs)
        values = torch.cat([pos[:n], self._processed_actions[:n], vel[:n], demand[:n], estimate[:n]], dim=1)
        rows = torch.cat([s.phase[:n, None].float(), values], dim=1).detach().cpu().tolist()
        with path.open("a", newline="") as stream:
            csv.writer(stream).writerows([env.common_step_counter, env._sim_step_counter, i, *row]
                                         for i, row in enumerate(rows))


class AutonomousLegPositionAction(JumpLegPositionAction):
    """Plain position targets; inheritance only reuses optional drive diagnostics."""

    def process_actions(self, actions):
        JointPositionAction.process_actions(self, actions)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._processed_actions.clamp_(min=limits[..., 0], max=limits[..., 1])


def autonomous_clearance_reward(env):
    state = get_state(env)
    # RewardManager multiplies by dt. A peak increment is a one-time event.
    return state.progress * (~env.termination_manager.terminated).float() / env.step_dt


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
