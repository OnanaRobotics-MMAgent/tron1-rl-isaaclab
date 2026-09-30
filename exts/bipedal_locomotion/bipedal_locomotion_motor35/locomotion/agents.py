from isaaclab.utils import configclass
from bipedal_locomotion_motor43.locomotion.agents import WF_TRON1AFlatPPORunnerCfg

@configclass
class Motor35LocomotionRunnerCfg(WF_TRON1AFlatPPORunnerCfg):
    experiment_name = "motor35_locomotion"
    save_interval = 100

