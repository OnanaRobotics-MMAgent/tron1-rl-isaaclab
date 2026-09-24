"""Small-tilt to overturned recovery for the current wheel-leg locomotion policy."""

from isaaclab.managers import EventTermCfg, RewardTermCfg
from isaaclab.utils import configclass

from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import WFBlindFlatEnvCfg
from .getup_env_cfg import GetUpCfg, GetUpCurriculumCfg, GetUpRewardsCfg, GetUpTerminationsCfg
from .mdp import rewards
from .mdp.actions import LimitedLegPositionAction
from .mdp.progressive_events import prepare_progressive, reset_progressive


@configclass
class WFProgressiveRecoveryEnvCfg(WFBlindFlatEnvCfg):
    getup: GetUpCfg = GetUpCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.scene.env_spacing = 6.0
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        # Preserve nominal joint offsets, scales (0.12 / 53.33), actuator gains,
        # observations and history of the source locomotion/standing checkpoint.
        # Larger raw leg actions can still reach the complete physical range.
        self.actions.joint_pos.class_type = LimitedLegPositionAction
        self.getup.initial_level = 0
        self.getup.tilt_ranges_deg = ((0., 5.),) + tuple(
            (float(max(0, high - 10)), float(high)) for high in range(10, 181, 5))
        self.getup.front_back_level = -1
        self.getup.overturned_level = -1
        self.getup.replay_probability = 0.20
        self.getup.min_curriculum_episodes = 4096
        self.getup.min_direction_episodes = 128
        self.getup.promote_success_rate = 0.80
        self.getup.promote_direction_success_rate = 0.70
        self.getup.promote_windows = 3
        self.getup.reset_clearance = 0.005
        self.getup.min_root_height = 0.0
        self.getup.height_tolerance = 0.04
        self.getup.hold_time = 2.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.debug_vis = False
        for name in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z'):
            setattr(self.commands.base_velocity.ranges, name, (0.0, 0.0))
        torque_soft = self.rewards.pen_wheel_torque_above_continuous
        self.rewards = GetUpRewardsCfg()
        self.rewards.upright.weight = 0.5
        self.rewards.height_progress = RewardTermCfg(func=rewards.height_progress, weight=6.0)
        self.rewards.low_height = RewardTermCfg(func=rewards.low_height_deficit, weight=-4.0)
        self.rewards.wheel_speed.weight = -5.0e-5
        self.rewards.wheel_continuous_torque = torque_soft
        self.terminations = GetUpTerminationsCfg()
        self.curriculum = GetUpCurriculumCfg()
        self.events.prepare_getup = EventTermCfg(func=prepare_progressive, mode='startup')
        self.events.reset_robot_base = EventTermCfg(func=reset_progressive, mode='reset')


@configclass
class WFProgressiveRecoveryEnvCfg_PLAY(WFProgressiveRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.getup.curriculum_enabled = False


@configclass
class WFInvertedRecoveryEnvCfg_PLAY(WFProgressiveRecoveryEnvCfg_PLAY):
    """Fixed 170–180 degree benchmark for repeatable torque playback."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.getup.initial_level = len(self.getup.tilt_ranges_deg) - 1
        self.getup.replay_probability = 0.0
        self.viewer.origin_type = 'asset_root'
        self.viewer.asset_name = 'robot'
        self.viewer.eye = (0.8, 0.8, 0.6)
        self.viewer.lookat = (0.0, 0.0, 0.12)


@configclass
class WFRecoveryLocomotionEnvCfg_PLAY(WFInvertedRecoveryEnvCfg_PLAY):
    """Continuous handoff scene: standing does not reset the recovered robot."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.terminations.success = None
        self.terminations.out_of_bounds = None  # locomotion may travel beyond the recovery circle
        self.curriculum.recovery = None
        # Some Kit builds resolve asset_root before Articulation data is initialized.
        # The dedicated play script follows the robot after the scene is ready.
        self.viewer.origin_type = 'env'
