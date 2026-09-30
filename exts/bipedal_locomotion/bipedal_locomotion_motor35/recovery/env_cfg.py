from isaaclab.utils import configclass
from bipedal_locomotion_motor43.recovery.progressive_env_cfg import WFProgressiveRecoveryEnvCfg
from ..common_cfg import configure_motor35

@configclass
class Motor35RecoveryEnvCfg(WFProgressiveRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_motor35(self, recovery=True)


@configclass
class Motor35RecoveryPlayEnvCfg(WFProgressiveRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_motor35(self, recovery=True, randomize=False)
        self.scene.num_envs = 1
        self.getup.curriculum_enabled = False
        self.getup.replay_probability = 0.
        self.getup.initial_level = len(self.getup.tilt_ranges_deg) - 1

