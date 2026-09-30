"""Separate Motor35 asset; never patches WF_TRON1A or its trained policies."""
from pathlib import Path
import hashlib
import json

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

from . import motor35_parameters as p

REPO = Path(__file__).resolve().parents[5]
PREPARED = Path(__file__).resolve().parents[1] / "urdf/robot.urdf"


def motor35_asset():
    """Load the independently generated asset after checking its source identity."""
    source = Path(__file__).resolve().parents[1] / "source/Motor35_URDF/轮腿总装/urdf/轮腿总装（角度限位调整版本）.urdf"
    cad = Path(__file__).resolve().parents[1] / "source/Motor35_URDF/转动惯量.txt"
    report_path = PREPARED.with_name("audit.json")
    if not PREPARED.exists() or not report_path.exists():
        raise FileNotFoundError("Run tools/prepare_motor35.py with the Isaac Sim Python environment first")
    report = json.loads(report_path.read_text())
    for path, key in ((source, "source_sha256"), (cad, "cad_sha256"), (PREPARED, "prepared_sha256"),
                      (Path(p.__file__), "profile_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[key]:
            raise ValueError(f"Motor35 source changed: {path}; rerun tools/prepare_motor35.py")
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(PREPARED), usd_dir=str(PREPARED.parent.parent / "usd/Motor35"),
            usd_file_name="Motor35.usd", fix_base=False, merge_fixed_joints=False,
            # Single hulls bridge gaps around the two hip motors and overlap thighs.
            collider_type="convex_decomposition", self_collision=True,
            replace_cylinders_with_capsules=False,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                target_type="none", gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=0.5,
                linear_damping=0., angular_damping=0.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., p.STAND_HEIGHT_M + .005), joint_pos=p.NOMINAL_JOINT_POS.copy(), joint_vel={".*": 0.}),
        soft_joint_pos_limit_factor=.9,
        actuators={
            # Linear torque-speed surrogate, NOT a measured curve/thermal model.
            # Effort limits are output-shaft limits; no extra 7x gear multiplier.
            "legs": DCMotorCfg(
                joint_names_expr=["(abad|hip|knee)_[LR]_Joint"],
                effort_limit=p.PEAK_TORQUE_NM, effort_limit_sim=p.PEAK_TORQUE_NM,
                saturation_effort=p.PEAK_TORQUE_NM,
                velocity_limit=p.NO_LOAD_SPEED_RAD_S, velocity_limit_sim=p.NO_LOAD_SPEED_RAD_S,
                stiffness=8., damping=.15, friction=0.),
            "wheels": DCMotorCfg(
                joint_names_expr=["wheel_[LR]_Joint"],
                effort_limit=p.PEAK_TORQUE_NM, effort_limit_sim=p.PEAK_TORQUE_NM,
                saturation_effort=p.PEAK_TORQUE_NM,
                velocity_limit=p.NO_LOAD_SPEED_RAD_S, velocity_limit_sim=p.NO_LOAD_SPEED_RAD_S,
                stiffness=0., damping=.02, friction=0.),
        },
    )
