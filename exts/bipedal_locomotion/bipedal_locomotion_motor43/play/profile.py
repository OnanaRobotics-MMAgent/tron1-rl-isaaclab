"""Asset and coordinate conventions of the verified Motor43 handoff."""
from pathlib import Path

TASK_ID = "Isaac-Motor43-Recovery-Locomotion-Play-v0"
FORWARD_AXIS = 1
LATERAL_AXIS = 0
SEPARATION_SIGN = 1.0  # right minus left in body-x
WHEEL_AXIS = 0
ACTION_SCALES = [.12] * 6 + [2 / .0375] * 2
WHEEL_SPEED_LIMIT = 60.0
COMMAND_LIMIT = 2.0
YAW_LIMIT = 3.141592653589793


def configure_asset(cfg, current_asset=False):
    assets = Path(__file__).resolve().parents[1] / "assets"
    if current_asset:
        return assets / "urdf/motor43.urdf", "current"
    # Exact asset/default pose of the supplied, validated policy pair. Bundling
    # this snapshot removes the old dependency on a commit in the original repo.
    assets = assets / "validated"
    urdf = assets / "urdf/motor43.urdf"
    cfg.scene.robot.spawn.usd_path = str(assets / "usd/WF_TRON1A/WF_TRON1A.usd")
    cfg.scene.robot.init_state.pos = (0., 0., .182)
    cfg.scene.robot.init_state.joint_pos = dict(
        abad_L_Joint=0., abad_R_Joint=0.,
        hip_L_Joint=.13437526, hip_R_Joint=-.13437526,
        knee_L_Joint=-.53190465, knee_R_Joint=.53190465,
        wheel_L_Joint=0., wheel_R_Joint=0.)
    if hasattr(cfg, 'getup'):
        cfg.getup.geometry_urdf = str(urdf)
    return urdf, "validated"
