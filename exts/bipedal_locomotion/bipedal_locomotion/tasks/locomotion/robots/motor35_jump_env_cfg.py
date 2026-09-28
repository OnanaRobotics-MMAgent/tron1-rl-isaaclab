"""Autonomous Motor35 clearance learning; assisted C2 mode is opt-in only."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.motor35_wheelfoot_cfg import MOTOR35_WHEEL_RADIUS_M
from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.mdp import motor35_jump as jump
from .motor35_wheelfoot_env_cfg import Motor35WFBlindFlatEnvCfg


@configclass
class JumpTerminationsCfg:
    # Manager execution order guarantees state is current before rewards.
    jump_update = DoneTerm(func=jump.JumpUpdate)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.20})
    minimum_height = DoneTerm(func=jump.jump_minimum_height, params={"minimum_height": .05})
    joint_limits = DoneTerm(func=mdp.joint_pos_out_of_limit, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=jump.LEG_NAMES, preserve_order=True)})


@configclass
class Motor35JumpSmallEnvCfg(Motor35WFBlindFlatEnvCfg):
    jump: object = jump.JumpCfg(autonomous=True)

    def __post_init__(self):
        super().__post_init__()
        # URDF FK at [-.3, .3, -.8, .8] gives grounded root z=0.203673 m.
        # The flat asset's 0.15 m spawn penetrates the floor by about 5 cm.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.206)
        self.episode_length_s = 20.0
        self.terminations = JumpTerminationsCfg()
        self.events.push_robot = None
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
        for name in ("policy", "obsHistory", "critic"):
            group = getattr(self.observations, name)
            group.jump_time = ObsTerm(func=jump.jump_time_obs)
            group.jump_clearance = ObsTerm(func=jump.jump_clearance_obs,
                params={"asset_cfg": wheel, "radius": MOTOR35_WHEEL_RADIUS_M})
            group.jump_vertical_speed = ObsTerm(func=jump.jump_vertical_speed_obs,
                params={"asset_cfg": SceneEntityCfg("robot")})

        self.configure_jump_learning(self.jump.autonomous)

    def configure_jump_learning(self, autonomous=True):
        self.jump.autonomous = autonomous
        self.actions.joint_pos.class_type = (
            jump.AutonomousLegPositionAction if autonomous else jump.JumpLegPositionAction)
        legs = SceneEntityCfg("robot", joint_names=jump.LEG_NAMES, preserve_order=True)

        for name in vars(self.rewards).copy():
            setattr(self.rewards, name, None)
        if autonomous:
            self.rewards.jump_clearance = RewTerm(func=jump.autonomous_clearance_reward, weight=16.)
            self.rewards.termination = RewTerm(func=mdp.is_terminated, weight=-100.)
            # Explicitly opt in only after stable jumping has been evaluated.
            self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=0.)
            self.rewards.action_smooth = RewTerm(func=jump.jump_reward, weight=0.,
                                                params={"kind": "action_smooth"})
            return
        # C2 overrides plus all non-overridden flat-task regularizers.
        weights = {
            "lin_vel_z": -.5, "base_height": 2.5, "action_smooth": -.001,
            "nominal_state": -.5, "track_lin_vel": 3., "track_ang_vel": 1., "track_heading": .25,
            "crouch": 7., "phase_action": 1.5, "thrust_pose": 5., "thrust_speed": 6.,
            "takeoff": 14., "takeoff_event": 80., "height": 5., "wheel_clearance": 16.,
            "airborne": 6., "symmetry": .5, "landing_pose": 6., "landing_soft": 100.,
            "landing_impact": -35., "recovery": 3., "success": 300., "failure": -120.,
        }
        for kind, weight in weights.items():
            setattr(self.rewards, "jump_" + kind, RewTerm(
                func=jump.jump_reward, weight=weight, params={"kind": kind}))
        self.rewards.ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-.05)
        self.rewards.orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-4.)
        self.rewards.dof_vel_legs = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": legs})
        self.rewards.dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5e-9)
        self.rewards.torques = RewTerm(func=mdp.joint_torques_l2, weight=-5e-5)
        self.rewards.dof_pos_limits = RewTerm(func=jump.jump_joint_margin, weight=-10., params={"asset_cfg": legs})
        self.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-.001)
        self.rewards.collision = RewTerm(func=mdp.undesired_contacts, weight=-5., params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"), "threshold": .1})
        self.rewards.leg_collision = RewTerm(func=mdp.undesired_contacts, weight=-2., params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*"]), "threshold": .1})
        self.rewards.leg_posture = RewTerm(func=jump.jump_leg_posture, weight=-.5, params={"asset_cfg": legs})
        self.rewards.termination = RewTerm(func=mdp.is_terminated, weight=-100.)


@configclass
class Motor35JumpSmallEnvCfg_PLAY(Motor35JumpSmallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8


@configclass
class Motor35JumpHighEnvCfg(Motor35JumpSmallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.jump.big_jump = True


@configclass
class Motor35JumpHighEnvCfg_PLAY(Motor35JumpHighEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
