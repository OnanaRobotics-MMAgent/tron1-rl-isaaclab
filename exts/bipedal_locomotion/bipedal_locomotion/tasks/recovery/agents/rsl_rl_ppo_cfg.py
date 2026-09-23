"""PPO configurations for recovery, reusing the WF network and encoder."""

from isaaclab.utils import configclass
from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import WF_TRON1AFlatPPORunnerCfg


@configclass
class WF_TRON1AGetUpPPORunnerCfg(WF_TRON1AFlatPPORunnerCfg):
    """Reuse the WF PPO, encoder and observation history without architecture changes."""

    experiment_name = "wf_tron_1a_getup"
    save_interval = 200


@configclass
class WF_TRON1AGetUpRecoveryPPORunnerCfg(WF_TRON1AGetUpPPORunnerCfg):
    experiment_name = "wf_tron_1a_getup_recovery"
    save_interval = 100

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.003
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.min_action_std = 0.03
        self.algorithm.max_action_std = 0.8


@configclass
class WF_TRON1AGetUpAutoPPORunnerCfg(WF_TRON1AGetUpRecoveryPPORunnerCfg):
    experiment_name = "wf_tron_1a_getup_auto"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.001


@configclass
class WF_TRON1AGetUpBoundedPPORunnerCfg(WF_TRON1AGetUpAutoPPORunnerCfg):
    experiment_name = "wf_tron_1a_getup_bounded"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.action_bound_loss_coef = 0.005
        # Filled by the runner from actual action scale/offset/clip and joint order.


@configclass
class WF_TRON1AFallenPPORunnerCfg(WF_TRON1AGetUpBoundedPPORunnerCfg):
    experiment_name = 'wf_tron_1a_fallen'


@configclass
class WF_TRON1AInvertedPPORunnerCfg(WF_TRON1AGetUpBoundedPPORunnerCfg):
    experiment_name = "wf_tron_1a_inverted"
