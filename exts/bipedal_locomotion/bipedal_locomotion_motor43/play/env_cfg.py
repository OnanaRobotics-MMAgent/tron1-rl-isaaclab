from isaaclab.utils import configclass
from ..recovery.progressive_env_cfg import WFInvertedRecoveryEnvCfg_PLAY

@configclass
class WFRecoveryLocomotionEnvCfg_PLAY(WFInvertedRecoveryEnvCfg_PLAY):
    """Continuous handoff scene: standing does not reset the recovered robot."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.terminations.success = None
        self.terminations.out_of_bounds = None  # locomotion may travel beyond the recovery circle
        self.curriculum.recovery = None
        # Some Kit builds resolve asset_root before Articulation data is initialized.
        # The dedicated play script follows the robot after the scene is ready.
        self.viewer.origin_type = 'env'

