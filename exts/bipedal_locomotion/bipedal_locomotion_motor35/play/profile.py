"""Motor35 handoff uses its own model, wheel axle and body-x command."""
from pathlib import Path
from ..assets.config import motor35_parameters as p

TASK_ID = "Isaac-Motor35-Recovery-Locomotion-Play-v0"
FORWARD_AXIS = 0
LATERAL_AXIS = 1
SEPARATION_SIGN = -1.0  # left minus right in body-y
WHEEL_AXIS = 1
ACTION_SCALES = [.5] * 6 + [p.WHEEL_ACTION_SCALE] * 2
WHEEL_SPEED_LIMIT = p.NO_LOAD_SPEED_RAD_S
COMMAND_LIMIT = p.MAX_COMMAND_SPEED_MPS
YAW_LIMIT = 1.5


def configure_asset(cfg, current_asset=False):
    return Path(__file__).resolve().parents[1] / "assets/urdf/robot.urdf", "current"
