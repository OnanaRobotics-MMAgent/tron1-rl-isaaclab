from isaaclab.utils import configclass
from bipedal_locomotion_motor43.recovery.agents import WFProgressiveRecoveryPPORunnerCfg

@configclass
class Motor35RecoveryRunnerCfg(WFProgressiveRecoveryPPORunnerCfg):
    experiment_name = "motor35_recovery"
    save_interval = 100

