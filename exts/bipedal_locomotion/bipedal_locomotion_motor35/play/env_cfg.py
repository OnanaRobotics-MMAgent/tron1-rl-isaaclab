"""Continuous Motor35 scene using the same handoff episode semantics."""
from isaaclab.utils import configclass
from ..recovery.env_cfg import Motor35RecoveryPlayEnvCfg


@configclass
class Motor35RecoveryLocomotionPlayEnvCfg(Motor35RecoveryPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.terminations.success = None
        self.terminations.out_of_bounds = None
        self.curriculum.recovery = None
        self.viewer.origin_type = 'env'
