"""Passive pose generation and recovery from a settled random-joint pose bank."""
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg
from .getup_env_cfg import WFGetUpAutoEnvCfg
from .mdp.fallen_events import prepare_fallen, reset_from_bank


@configclass
class FallenPoseCfg:
    bank_path: str = ''
    split: str = 'train'
    old_pose_probability: float = 0.30


@configclass
class WFFallenPoseGenerationEnvCfg(WFGetUpAutoEnvCfg):
    """No active position control: gravity, damping and joint limits settle the robot."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 128
        self.scene.robot.actuators['legs'].stiffness = 0.0
        self.scene.robot.actuators['legs'].damping = 2.5
        self.getup.curriculum_enabled = False
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        self.events.robot_physics_material.params.update(
            static_friction_range=(0.8, 0.8), dynamic_friction_range=(0.7, 0.7))


@configclass
class WFFallenRecoveryEnvCfg(WFGetUpAutoEnvCfg):
    fallen: FallenPoseCfg = FallenPoseCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        # These 3 levels index joint-deformation bands, not body tilt ranges.
        # Ranges here are used only for neutral-joint rehearsal resets.
        self.getup.tilt_ranges_deg = ((0., 180.),) * 3
        self.getup.initial_level = 0
        self.getup.replay_probability = 0.0
        self.getup.promote_success_rate = 0.90
        self.getup.promote_direction_success_rate = 0.80
        self.events.prepare_getup = EventTermCfg(func=prepare_fallen, mode='startup')
        self.events.reset_robot_base = EventTermCfg(func=reset_from_bank, mode='reset')
        # Keep the previous reward definition for a controlled baseline comparison.


@configclass
class WFFallenRecoveryEnvCfg_PLAY(WFFallenRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.getup.curriculum_enabled = False
        self.fallen.split = 'test'
        self.fallen.old_pose_probability = 0.0
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
