"""In-place Motor35 small/high jump tasks."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.motor35_wheelfoot_cfg import MOTOR35_WHEEL_RADIUS_M
from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.mdp import motor35_jump as jump
from .motor35_wheelfoot_env_cfg import Motor35WFBlindFlatEnvCfg


@configclass
class Motor35JumpSmallEnvCfg(Motor35WFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 2.0
        self.actions.joint_pos.class_type = jump.JumpLegPositionAction
        self.actions.joint_pos.preserve_order = True
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = False
        wheel = SceneEntityCfg("robot", body_names=["wheel_L_Link", "wheel_R_Link"])
        contact = SceneEntityCfg("contact_forces", body_names=["wheel_L_Link", "wheel_R_Link"])
        for name in ("policy", "obsHistory", "critic"):
            group = getattr(self.observations, name)
            group.jump_time = ObsTerm(func=jump.jump_time_obs)
            group.jump_clearance = ObsTerm(
                func=jump.jump_clearance_obs, params={"asset_cfg": wheel, "radius": MOTOR35_WHEEL_RADIUS_M})
            group.jump_vertical_speed = ObsTerm(
                func=jump.jump_vertical_speed_obs, params={"asset_cfg": SceneEntityCfg("robot")})

        for name in vars(self.rewards).copy():
            setattr(self.rewards, name, None)
        self.rewards.jump_outcome = RewTerm(
            func=jump.JumpOutcome, weight=15.0,
            params={"asset_cfg": wheel, "sensor_cfg": contact,
                    "radius": MOTOR35_WHEEL_RADIUS_M, "target": 0.17, "big_jump": False})
        self.rewards.takeoff_velocity = RewTerm(
            func=jump.jump_takeoff_velocity, weight=2.0,
            params={"asset_cfg": wheel, "sensor_cfg": contact, "radius": MOTOR35_WHEEL_RADIUS_M})
        self.rewards.stationary = RewTerm(
            func=jump.jump_stationary, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")})
        self.rewards.wheel_speed = RewTerm(
            func=jump.jump_wheel_speed, weight=-0.0001,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_L_Joint", "wheel_R_Joint"])})
        self.rewards.upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
        self.rewards.joint_limits = RewTerm(
            func=mdp.joint_pos_limits, weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")})
        # The reference trajectory supplies dense shaping when the 3 Nm motors
        # cannot produce flight immediately at the start of training.
        self.rewards.reference = RewTerm(
            func=jump.jump_reference_error, weight=-0.5,
            params={"asset_cfg": SceneEntityCfg(
                "robot", preserve_order=True, joint_names=[
                    "abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
                    "knee_L_Joint", "knee_R_Joint"])})


@configclass
class Motor35JumpSmallEnvCfg_PLAY(Motor35JumpSmallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8


@configclass
class Motor35JumpHighEnvCfg(Motor35JumpSmallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.jump_outcome.params["big_jump"] = True
        self.rewards.jump_outcome.params["target"] = 0.17


@configclass
class Motor35JumpHighEnvCfg_PLAY(Motor35JumpHighEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
