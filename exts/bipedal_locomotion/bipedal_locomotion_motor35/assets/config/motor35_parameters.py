"""Motor35 source conventions and output-shaft ratings (no simulator imports).

Ratings: DAMIAO 2025 selection catalog, p.6, DM-J3507-2EC, 24 V.
https://www.dmbot.cn/index.php?c=show&id=118
The internal 7:1 reduction is ALREADY included; do not multiply torque again.
Controller gains, command ranges and randomization ranges are simulation
starting points, not measured hardware characteristics. See docs/motor35.md.
"""

import math

MODEL = "DM-J3507-2EC"
VOLTAGE_V = 24.0
RATED_TORQUE_NM = 0.8
PEAK_TORQUE_NM = 3.0
NO_LOAD_SPEED_RAD_S = 460.0 * 2.0 * math.pi / 60.0
RATED_SPEED_RAD_S = 150.0 * 2.0 * math.pi / 60.0
RATED_POWER_W = 12.0
INTERNAL_GEAR_RATIO = 7.0
# Derived from STL extents about the wheel axle, not the old 37.5 mm wheel.
WHEEL_RADIUS_M = 0.04
WHEEL_WIDTH_M = 0.032
WHEEL_CENTER_Y_M = 0.010
MAX_COMMAND_SPEED_MPS = 0.5
WHEEL_ACTION_SCALE = RATED_SPEED_RAD_S

JOINT_MAP = {
    "joint_hip_R": "abad_R_Joint", "Joint_hip_L": "abad_L_Joint",
    "Joint_thigh_R": "hip_R_Joint", "Joint_thigh_L": "hip_L_Joint",
    "Joint_calf_R": "knee_R_Joint", "Joint_calf_L": "knee_L_Joint",
    "Joint_wheel_R": "wheel_R_Joint", "Joint_wheel_L": "wheel_L_Joint",
}
LINK_MAP = {
    "base_link": "base_Link",
    "link_hip_R": "abad_R_Link", "Link_hip_L": "abad_L_Link",
    "link_thigh_R": "hip_R_Link", "Link_thigh_L": "hip_L_Link",
    "Link_calf_R": "knee_R_Link", "Link_calf_L": "knee_L_Link",
    "Link_wheel_R": "wheel_R_Link", "Link_wheel_L": "wheel_L_Link",
}
# FK solution: hip safely inside soft limits, whole-robot COM over wheel axle
# in body-x. This is a reference pose, not proof of dynamic balance.
NOMINAL_JOINT_POS = {
    "abad_L_Joint": 0.0, "abad_R_Joint": 0.0,
    "hip_L_Joint": -0.08, "hip_R_Joint": 0.08,
    "knee_L_Joint": -0.65901083546, "knee_R_Joint": 0.65901083546,
    "wheel_L_Joint": 0.0, "wheel_R_Joint": 0.0,
}
STAND_HEIGHT_M = 0.1877140561
