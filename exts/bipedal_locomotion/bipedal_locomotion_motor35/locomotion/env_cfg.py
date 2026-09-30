from isaaclab.utils import configclass
from bipedal_locomotion_motor43.locomotion.env_cfg import WFBlindFlatEnvCfg
from ..common_cfg import configure_motor35

@configclass
class Motor35LocomotionEnvCfg(WFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_motor35(self)


@configclass
class Motor35LocomotionPlayEnvCfg(WFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_motor35(self, randomize=False)
        self.scene.num_envs = 1

