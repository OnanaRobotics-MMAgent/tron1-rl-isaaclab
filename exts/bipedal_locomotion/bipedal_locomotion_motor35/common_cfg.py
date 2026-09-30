"""New-model tasks share motor limits, interface and measured-source geometry.

The DR intervals and controller gains are initial engineering choices. Run the
preflight and a short pilot before full training; see docs/motor35.md.
"""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion_motor35.assets.config.motor35_cfg import motor35_asset, PREPARED
from bipedal_locomotion_motor35.assets.config import motor35_parameters as p
from bipedal_locomotion_motor43.locomotion import mdp as loco
from bipedal_locomotion_motor43.locomotion.env_cfg import WFBlindFlatEnvCfg
from bipedal_locomotion_motor43.recovery.progressive_env_cfg import WFProgressiveRecoveryEnvCfg
from bipedal_locomotion_motor43.recovery.mdp.actions import LimitedLegPositionAction
from . import mdp


def configure_motor35(cfg, recovery=False, randomize=True):
    cfg.scene.robot = motor35_asset()
    cfg.policy_contract = {
        "robot": "Motor35", "interface_version": 1,
        "prepared_urdf_sha256": hashlib.sha256(PREPARED.read_bytes()).hexdigest(),
        "motor_config_sha256": hashlib.sha256((Path(p.__file__).parent / "motor35_cfg.py").read_bytes()).hexdigest(),
        "leg_action_scale": .5, "wheel_action_scale": p.WHEEL_ACTION_SCALE,
        "physics_dt": .0025, "policy_dt": .02,
    }
    # Explicit motor torques need smaller substeps than the old implicit PD.
    # Keep policy control and observation history at the same 50 Hz.
    cfg.sim.dt = .0025
    cfg.decimation = 8
    cfg.sim.render_interval = 16
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    cfg.actions.joint_pos.class_type = LimitedLegPositionAction
    cfg.actions.joint_pos.scale = .5
    cfg.actions.joint_pos.clip = None
    cfg.actions.joint_vel.scale = p.WHEEL_ACTION_SCALE
    cfg.actions.joint_vel.clip = {".*": (-p.NO_LOAD_SPEED_RAD_S, p.NO_LOAD_SPEED_RAD_S)}

    # Reset legacy DR explicitly; toggling deterministic_baseline=False would
    # reintroduce duplicated mass scaling, absolute gains and 500 N pushes.
    for name in ("add_base_mass", "add_link_mass", "radomize_rigid_body_mass_inertia",
                 "robot_joint_stiffness_and_damping", "robot_center_of_mass",
                 "randomize_actuator_gains", "push_robot", "reset_robot_joints"):
        setattr(cfg.events, name, None)
    if randomize:
        cfg.events.add_link_mass = EventTerm(func=lab_mdp.randomize_rigid_body_mass, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (.9, 1.1), "operation": "scale", "recompute_inertia": True})
        cfg.events.robot_center_of_mass = EventTerm(func=loco.randomize_rigid_body_coms, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "com_distribution_params": ((-.003, .003),) * 3, "operation": "add"})
        cfg.events.randomize_actuator_gains = EventTerm(func=lab_mdp.randomize_actuator_gains, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (.9, 1.1), "damping_distribution_params": (.9, 1.1),
            "operation": "scale"})  # scaling keeps wheel Kp exactly zero
    for name, static, dynamic in (
        ("robot_physics_material", (.6, 1.0), (.5, .9)),
        ("wheel_physics_material", (.7, 1.2), (.6, 1.1)),
    ):
        cfg_term = getattr(cfg.events, name)
        if not randomize:
            static = (sum(static) / 2,) * 2
            dynamic = (sum(dynamic) / 2,) * 2
        cfg_term.params.update(static_friction_range=static, dynamic_friction_range=dynamic,
                               restitution_range=(0., .05) if randomize else (0., 0.),
                               num_buckets=64 if randomize else 1, make_consistent=True)
    cfg.scene.terrain.physics_material.restitution = 0.
    # No repeated CPU PhysX queries, default-mass leakage or collider-count-dependent critic width.
    for name in ("robot_mass", "robot_inertia", "robot_joint_stiffness", "robot_joint_damping", "robot_material_properties"):
        setattr(cfg.observations.critic, name, None)
    cfg.observations.critic.motor35_properties = ObsTerm(func=mdp.physical_properties)
    cfg.events.capture_motor35_properties = EventTerm(func=mdp.capture_physical_properties, mode="startup")
    for name in ("policy", "obsHistory"):
        group = getattr(cfg.observations, name)
        group.enable_corruption = randomize
        group.last_action.noise = None  # software commands are known, not noisy sensors
    cfg.observations.critic.enable_corruption = False
    cfg.observations.commands.enable_corruption = False

    # Same physical bound and soft rated-torque target for ALL eight new motors.
    # A soft reward does not impose a thermal/overload-time guarantee.
    cfg.rewards.motor_rated_torque = RewTerm(func=loco.joint_torque_excess_l1, weight=-.25, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "continuous_torque": p.RATED_TORQUE_NM})
    if recovery:
        cfg.rewards.wheel_continuous_torque = None
        cfg.getup.geometry_urdf = str(PREPARED)
        cfg.getup.leg_joint_limits = {
            joint.get("name"): (float(joint.find("limit").get("lower")), float(joint.find("limit").get("upper")))
            for joint in ET.parse(PREPARED).getroot().findall("joint") if joint.get("type") == "revolute"
        }
        cfg.getup.target_height = p.STAND_HEIGHT_M
        cfg.getup.height_tolerance = .025
        cfg.getup.min_wheel_force = 2.
        cfg.getup.max_body_force = 2.
        cfg.getup.landing_force = 2.
        # About 4x robot weight: initial penalty threshold, not a hardware rating.
        cfg.rewards.impact.params["threshold"] = 100.
    else:
        cfg.commands.base_velocity.heading_command = False
        cfg.commands.base_velocity.rel_heading_envs = 0.
        cfg.commands.base_velocity.rel_standing_envs = .2
        cfg.commands.base_velocity.debug_vis = False
        cfg.commands.base_velocity.ranges.lin_vel_x = (-p.MAX_COMMAND_SPEED_MPS, p.MAX_COMMAND_SPEED_MPS)
        cfg.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
        cfg.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        cfg.rewards.pen_base_height.params["target_height"] = p.STAND_HEIGHT_M
        cfg.rewards.pen_abad_torque_excess = None
        cfg.rewards.pen_wheel_torque_above_continuous = None
        cfg.rewards.pen_body_y_velocity_error.func = mdp.forward_velocity_error
        cfg.rewards.pen_body_y_velocity_error.params = {}
        cfg.rewards.pen_reverse_motion.func = mdp.reverse_motion
        cfg.rewards.pen_reverse_motion.params = {}
        cfg.rewards.rew_same_foot_x_position.params["axis_idx"] = 0
        cfg.rewards.pen_feet_distance.params.update(min_feet_distance=.14, max_feet_distance=.20)


