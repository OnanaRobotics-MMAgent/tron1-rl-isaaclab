import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.motor35_wheelfoot_cfg import (
    MOTOR35_MAX_BODY_SPEED_MPS,
    MOTOR35_WHEEL_RADIUS_M,
    MOTOR35_WHEEL_TARGET_SPEED_RAD_S,
    MOTOR35_WHEELFOOT_CFG,
)
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import WFBlindFlatEnvCfg


def body_x_velocity_error(env, command_name, asset_cfg, scale=1.0):
    target = env.command_manager.get_command(command_name)[:, 0]
    actual = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    return torch.sqrt(1.0 + ((target - actual) / scale).square()) - 1.0


def body_x_reverse_motion(env, command_name, asset_cfg, deadband=0.1):
    target = env.command_manager.get_command(command_name)[:, 0]
    actual = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    return torch.relu(-torch.sign(target) * actual) * (target.abs() > deadband)


@configclass
class Motor35WFBlindFlatEnvCfg(WFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # The converted URDF retains its frames and joint axes; only the
        # link/joint names and mesh paths are adapted to the WF task.
        self.scene.robot = MOTOR35_WHEELFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.base_velocity.ranges.lin_vel_x = (
            -MOTOR35_MAX_BODY_SPEED_MPS, MOTOR35_MAX_BODY_SPEED_MPS
        )
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.actions.joint_vel.scale = MOTOR35_WHEEL_TARGET_SPEED_RAD_S
        self.events.prepare_quantity_for_tron1_piper.params["foot_radius"] = MOTOR35_WHEEL_RADIUS_M

        self.rewards.pen_base_height.params["target_height"] = 0.14
        self.rewards.pen_abad_torque_excess = None
        self.rewards.pen_wheel_torque_above_continuous = None
        self.rewards.rew_same_foot_x_position.params["axis_idx"] = 0
        self.rewards.pen_body_y_velocity_error = None
        self.rewards.pen_reverse_motion = None
        self.rewards.pen_body_x_velocity_error = RewTerm(
            func=body_x_velocity_error, weight=-1.0,
            params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.pen_body_x_reverse_motion = RewTerm(
            func=body_x_reverse_motion, weight=-1.0,
            params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class Motor35WFBlindFlatEnvCfg_PLAY(Motor35WFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
