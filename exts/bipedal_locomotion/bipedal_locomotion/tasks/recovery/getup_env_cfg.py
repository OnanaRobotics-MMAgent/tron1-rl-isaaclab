"""WF_TRON1A flat-ground recovery, using the existing observation/actuator stack."""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from bipedal_locomotion.tasks.locomotion import mdp as locomotion_mdp
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import WFBlindFlatEnvCfg

from . import mdp
from .mdp.actions import LimitedLegPositionAction
from .mdp.limits import LEG_JOINT_LIMITS


def success_entities():
    return {
        "wheel_cfg": SceneEntityCfg("contact_forces", body_names="wheel_[LR]_Link"),
        "body_cfg": SceneEntityCfg("contact_forces", body_names=["base_Link", "(abad|hip|knee)_[LR]_Link"]),
        "leg_cfg": SceneEntityCfg("robot", joint_names="(abad|hip|knee)_[LR]_Joint"),
    }


@configclass
class GetUpCfg:
    tilt_ranges_deg: tuple = ((0.0, 20.0), (20.0, 65.0), (80.0, 100.0), (80.0, 110.0))
    initial_level: int = 0
    curriculum_enabled: bool = True
    front_back_level: int = 2
    overturned_level: int = 3
    promote_windows: int = 1
    promote_direction_success_rate: float = 0.0
    min_direction_episodes: int = 128
    replay_probability: float = 0.2
    min_curriculum_episodes: int = 1024
    promote_success_rate: float = 0.75
    # The supplied wheel mesh has a radius of about 0.0375 m and the neutral
    # wheel bottom is about 0.16 m below base_Link.
    target_height: float = 0.18
    reset_clearance: float = 0.02
    success_tilt: float = math.radians(15.0)
    height_tolerance: float = 0.07
    max_linear_speed: float = 0.25
    max_angular_speed: float = 0.5
    min_wheel_force: float = 5.0
    max_body_force: float = 5.0
    max_pose_error: float = 0.16  # mean squared leg joint error (rad^2)
    hold_time: float = 1.0
    min_episode_time: float = 0.5
    max_drift: float = 2.5
    leg_joint_limits: dict = LEG_JOINT_LIMITS.copy()
    joint_limit_tolerance: float = 0.05  # allow small solver overshoot, never a full turn


@configclass
class GetUpRewardsCfg:
    upright = RewTerm(func=mdp.upright, weight=2.0)
    height = RewTerm(func=mdp.height_tracking, weight=4.0)
    support = RewTerm(func=mdp.wheel_support, weight=2.0,
                      params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_[LR]_Link")})
    nominal_pose = RewTerm(func=mdp.nominal_pose, weight=1.0,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names="(abad|hip|knee)_[LR]_Joint")})
    quiet = RewTerm(func=mdp.quiet_standing, weight=2.0)
    stable = RewTerm(func=mdp.stable_reward, weight=5.0, params=success_entities())
    # Exceeds the maximum discounted dense return (~32 at gamma=.99, dt=.02),
    # so avoiding termination to harvest standing rewards is not advantageous.
    success = RewTerm(func=mdp.success_bonus, weight=50.0)
    torque = RewTerm(func=locomotion_mdp.joint_torques_l2, weight=-2.0e-5)
    joint_acc = RewTerm(func=locomotion_mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate = RewTerm(func=locomotion_mdp.action_rate_l2, weight=-0.01)
    joint_limits = RewTerm(func=locomotion_mdp.joint_pos_limits, weight=-2.0,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names="(abad|hip|knee)_[LR]_Joint")})
    impact = RewTerm(func=mdp.excessive_impact, weight=-0.1,
                    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 600.0})
    wheel_speed = RewTerm(func=mdp.wheel_speed_near_stand, weight=-0.002,
                         params={"asset_cfg": SceneEntityCfg("robot", joint_names="wheel_[LR]_Joint")})
    leg_limit_failure = RewTerm(func=mdp.leg_limit_failure, weight=-10.0,
                               params={"asset_cfg": SceneEntityCfg("robot", joint_names="(abad|hip|knee)_[LR]_Joint")})


@configclass
class GetUpTerminationsCfg:
    time_out = DoneTerm(func=locomotion_mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.sustained_success, params=success_entities())
    out_of_bounds = DoneTerm(func=mdp.outside_recovery_area)
    leg_limit_violation = DoneTerm(func=mdp.leg_limit_violation,
                                  params={"leg_cfg": SceneEntityCfg("robot", joint_names="(abad|hip|knee)_[LR]_Joint")})


@configclass
class GetUpCurriculumCfg:
    recovery = CurrTerm(func=mdp.recovery_levels)


@configclass
class WFGetUpEnvCfg(WFBlindFlatEnvCfg):
    # Recovery retains its own random fallen-pose curriculum and material
    # settings; the deterministic baseline is for flat locomotion only.
    deterministic_baseline: bool = False
    getup: GetUpCfg = GetUpCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 8.0
        self.scene.env_spacing = 6.0
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.4)
        self.scene.terrain.physics_material.restitution = 0.0
        self.rewards = GetUpRewardsCfg()
        self.terminations = GetUpTerminationsCfg()
        self.curriculum = GetUpCurriculumCfg()

        # Same 6 position + 2 velocity actions, wider reach for recovery. Keep the
        # Use the conservative actuator limits from the adapted wheel-leg USD.
        self.actions.joint_pos.scale = 1.0
        self.actions.joint_pos.class_type = LimitedLegPositionAction
        self.actions.joint_pos.clip = {
            "abad_L_Joint": (-0.29, 1.30), "abad_R_Joint": (-1.30, 0.29),
            "hip_L_Joint": (-0.95, 1.30), "hip_R_Joint": (-1.30, 0.95),
            "knee_L_Joint": (-0.80, 1.29), "knee_R_Joint": (-1.29, 0.80),
        }
        self.actions.joint_vel.scale = 2.0
        self.actions.joint_vel.clip = {".*": (-20.0, 20.0)}
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # In particular, the locomotion gain randomizer must not turn the wheels
        # into position servos by assigning non-zero stiffness to every joint.
        for name in ("add_base_mass", "add_link_mass", "radomize_rigid_body_mass_inertia",
                     "robot_joint_stiffness_and_damping", "robot_center_of_mass",
                     "randomize_actuator_gains", "push_robot", "reset_robot_joints"):
            setattr(self.events, name, None)
        self.events.robot_physics_material.params.update(
            static_friction_range=(0.7, 1.0), dynamic_friction_range=(0.6, 0.7),
            restitution_range=(0.0, 0.0),
        )
        self.events.prepare_getup = EventTerm(func=mdp.prepare_getup, mode="startup")
        self.events.reset_robot_base = EventTerm(func=mdp.reset_fallen, mode="reset")


@configclass
class WFGetUpEnvCfg_PLAY(WFGetUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.getup.initial_level = 3
        self.getup.curriculum_enabled = False
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        self.events.robot_physics_material.params.update(
            static_friction_range=(0.8, 0.8), dynamic_friction_range=(0.7, 0.7),
        )


@configclass
class WFGetUpRecoveryEnvCfg(WFGetUpEnvCfg):
    """Short recovery branch: overlap stage 0 and stage 1 before larger tilts."""

    def __post_init__(self):
        super().__post_init__()
        self.getup.initial_level = 1
        self.getup.curriculum_enabled = False
        self.getup.tilt_ranges_deg = ((0.0, 20.0), (15.0, 25.0), (80.0, 100.0), (80.0, 110.0))
        self.rewards.height_progress = RewTerm(func=mdp.height_progress, weight=3.0)


@configclass
class WFGetUpRecoveryEnvCfg_PLAY(WFGetUpRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False


@configclass
class WFGetUpAutoEnvCfg(WFGetUpRecoveryEnvCfg):
    """Conservative automatic progression with four-direction promotion checks."""

    def __post_init__(self):
        super().__post_init__()
        self.getup.tilt_ranges_deg = ((0., 20.), (15., 25.)) + tuple(
            (float(high - 10), float(high)) for high in range(30, 181, 5))
        self.getup.curriculum_enabled = True
        # Every level uses front/back/left/right; no hard-coded stage-2/3 jumps.
        self.getup.front_back_level = -1
        self.getup.overturned_level = -1
        self.getup.min_curriculum_episodes = 8192
        self.getup.promote_success_rate = 0.80
        self.getup.promote_direction_success_rate = 0.70
        self.getup.promote_windows = 3
        self.getup.replay_probability = 0.20
        self.rewards.upright.weight = 0.5
        self.rewards.height_progress.weight = 6.0
        self.rewards.low_height = RewTerm(func=mdp.low_height_deficit, weight=-4.0)
        # Max positive dense reward is 20.5/s; bonus remains above its
        # discounted sum (41 at gamma=.99 and dt=.02).


@configclass
class WFGetUpAutoEnvCfg_PLAY(WFGetUpAutoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.getup.curriculum_enabled = False
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
