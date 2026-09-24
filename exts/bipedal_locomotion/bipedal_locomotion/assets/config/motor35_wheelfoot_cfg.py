import os

from isaaclab.actuators import ImplicitActuatorCfg

from bipedal_locomotion.assets.config.wheelfoot_cfg import WHEELFOOT_CFG


_ASSET_DIR = os.path.dirname(__file__)
_USD_PATH = os.path.join(_ASSET_DIR, "../usd/Motor35_WF/Motor35_WF.usd")

MOTOR35_WHEEL_RADIUS_M = 0.04
MOTOR35_MAX_BODY_SPEED_MPS = 1.0
MOTOR35_WHEEL_TARGET_SPEED_RAD_S = MOTOR35_MAX_BODY_SPEED_MPS / MOTOR35_WHEEL_RADIUS_M
MOTOR35_EFFORT_LIMIT_NM = 3.0
MOTOR35_VELOCITY_LIMIT_RAD_S = 48.171087

MOTOR35_WHEELFOOT_CFG = WHEELFOOT_CFG.replace(
    spawn=WHEELFOOT_CFG.spawn.replace(usd_path=_USD_PATH),
    init_state=WHEELFOOT_CFG.init_state.replace(
        pos=(0.0, 0.0, 0.15),
        joint_pos={
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": -0.3,
            "hip_R_Joint": 0.3,
            "knee_L_Joint": -0.8,
            "knee_R_Joint": 0.8,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "abad_L_Joint", "abad_R_Joint",
                "hip_L_Joint", "hip_R_Joint",
                "knee_L_Joint", "knee_R_Joint",
            ],
            effort_limit_sim=MOTOR35_EFFORT_LIMIT_NM,
            velocity_limit_sim=MOTOR35_VELOCITY_LIMIT_RAD_S,
            stiffness={"abad_.*": 8.0, "hip_.*": 12.0, "knee_.*": 12.0},
            damping=0.8,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_L_Joint", "wheel_R_Joint"],
            effort_limit_sim=MOTOR35_EFFORT_LIMIT_NM,
            velocity_limit_sim=MOTOR35_VELOCITY_LIMIT_RAD_S,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
        ),
    },
)
