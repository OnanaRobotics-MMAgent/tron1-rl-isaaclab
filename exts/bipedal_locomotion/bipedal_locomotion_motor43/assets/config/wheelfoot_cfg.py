import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

current_dir = os.path.dirname(__file__)
usd_path = os.path.join(current_dir, "../usd/WF_TRON1A/WF_TRON1A.usd")

# Calibrated wheel-side limits for the requested 1:1 direct-drive assumption.
# The continuous limit is enforced softly by the training reward; the peak
# limit is the hard per-step safety bound sent to the physics solver.
WHEEL_RADIUS_M = 0.0375
WHEEL_MAX_BODY_SPEED_MPS = 2.0
WHEEL_TARGET_SPEED_RAD_S = WHEEL_MAX_BODY_SPEED_MPS / WHEEL_RADIUS_M
WHEEL_SPEED_LIMIT_RAD_S = 60.0
WHEEL_CONTINUOUS_TORQUE_NM = 0.8
WHEEL_PEAK_TORQUE_NM = 2.0

WHEELFOOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=0.5,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),
        joint_pos={
            ".*_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "abad_L_Joint",
                "abad_R_Joint",
                "hip_L_Joint",
                "hip_R_Joint",
                "knee_L_Joint",
                "knee_R_Joint",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=12.0,
            stiffness={"abad_.*": 8.0, "hip_.*": 12.0, "knee_.*": 12.0},
            damping=0.8,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
                "wheel_L_Joint",
                "wheel_R_Joint",
            ],
            effort_limit_sim=WHEEL_PEAK_TORQUE_NM,
            velocity_limit_sim=WHEEL_SPEED_LIMIT_RAD_S,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
        ), # TODO: change to delayed implicit actuator
    },
)
