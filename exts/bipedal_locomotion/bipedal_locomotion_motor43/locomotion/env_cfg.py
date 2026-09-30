import math

from isaaclab.utils import configclass

from bipedal_locomotion_motor43.assets.config.wheelfoot_cfg import WHEELFOOT_CFG
from bipedal_locomotion_motor43.locomotion.base_env_cfg import WFEnvCfg


from isaaclab.sensors import RayCasterCfg, patterns
from bipedal_locomotion_motor43.locomotion import mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm


######################
# Wheelfoot Base Environment
######################


@configclass
class WFBaseEnvCfg(WFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = WHEELFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"
        
        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class WFBaseEnvCfg_PLAY(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


############################
# Wheelfoot Blind Flat Environment
############################


def configure_flat_baseline(cfg):
    """Apply the no-randomization baseline only to the flat locomotion task."""
    for name in (
        "add_base_mass", "add_link_mass", "radomize_rigid_body_mass_inertia",
        "robot_joint_stiffness_and_damping", "robot_center_of_mass",
        "randomize_actuator_gains", "push_robot", "reset_robot_joints",
    ):
        setattr(cfg.events, name, None)

    # Assign one fixed non-wheel material. Preserve the separately calibrated
    # unit wheel friction and the ground's multiply combine mode.
    cfg.events.robot_physics_material.params.update(
        asset_cfg=SceneEntityCfg("robot", body_names="(?!wheel_).+"),
        static_friction_range=(0.8, 0.8),
        dynamic_friction_range=(0.7, 0.7),
        restitution_range=(0.0, 0.0),
        num_buckets=1,
        make_consistent=True,
    )
    cfg.events.wheel_physics_material.params.update(
        static_friction_range=(1.0, 1.0),
        dynamic_friction_range=(1.0, 1.0),
        restitution_range=(0.0, 0.0),
        num_buckets=1,
        make_consistent=True,
    )
    cfg.scene.terrain.physics_material.restitution = 0.0
    cfg.events.reset_robot_base = EventTerm(
        func=mdp.reset_scene_to_default, mode="reset",
        params={"reset_joint_targets": True},
    )
    for name in ("policy", "critic", "commands", "obsHistory"):
        getattr(cfg.observations, name).enable_corruption = False

    # With hip_L <= 0 and hip_R >= 0, this mirrored pose keeps both hips
    # inside the soft limits and starts the wheels just above the ground.
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.201)
    cfg.scene.robot.init_state.joint_pos = {
        "abad_L_Joint": 0.0, "abad_R_Joint": 0.0,
        "hip_L_Joint": -0.055, "hip_R_Joint": 0.055,
        "knee_L_Joint": -0.75, "knee_R_Joint": 0.75,
        "wheel_L_Joint": 0.0, "wheel_R_Joint": 0.0,
    }
    # Use the actual squared-height term rather than the legacy absolute-error
    # helper. The target is grounded height, not the spawn clearance.
    cfg.rewards.pen_base_height.func = mdp.base_height_l2
    cfg.rewards.pen_base_height.params["target_height"] = 0.18
    cfg.rewards.pen_joint_vel_wheel_l2.weight = -5.0e-4


def configure_flat_tracking_rewards(cfg):
    """Reward changes for body-y wheeled flat locomotion only, including play."""
    # At 2 m/s and r=0.0375 m this costs 0.284, not 2.844 (tracking pays 3).
    cfg.rewards.pen_joint_vel_wheel_l2.weight = -5.0e-5
    cfg.rewards.pen_body_y_velocity_error = RewTerm(
        func=mdp.body_y_velocity_error, weight=-1.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"), "scale": 1.0},
    )
    cfg.rewards.pen_reverse_motion = RewTerm(
        func=mdp.body_y_reverse_motion, weight=-1.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"), "deadband": 0.1},
    )
    # Replace the weak foot-y bonus rather than stack another symmetry term.
    cfg.rewards.rew_leg_symmetry = RewTerm(
        func=mdp.joint_mirror_pose_l2, weight=-2.0,
        params={"command_name": "base_velocity", "turn_rate": 0.5,
                "asset_cfg": SceneEntityCfg("robot", preserve_order=True, joint_names=[
                    "abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
                    "knee_L_Joint", "knee_R_Joint"])},
    )
    cfg.rewards.rew_same_foot_x_position = RewTerm(
        func=mdp.straight_feet_alignment, weight=-50.0,
        params={"command_name": "base_velocity", "turn_rate": 0.5, "axis_idx": 1,
                "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")},
    )


@configclass
class WFBlindFlatEnvCfg(WFBaseEnvCfg):
    deterministic_baseline: bool = True

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None

        if self.deterministic_baseline:
            configure_flat_baseline(self)
        configure_flat_tracking_rewards(self)


@configclass
class WFBlindFlatEnvCfg_PLAY(WFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


