"""Explicit task registrations for this robot; no legacy aliases."""
import gymnasium as gym

for name, config, runner in (
    ('Isaac-Motor43-GetUp-v0', 'bipedal_locomotion_motor43.recovery.getup_env_cfg:WFGetUpEnvCfg', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AGetUpPPORunnerCfg'),
    ('Isaac-Motor43-GetUp-Play-v0', 'bipedal_locomotion_motor43.recovery.getup_env_cfg:WFGetUpEnvCfg_PLAY', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AGetUpPPORunnerCfg'),
    ('Isaac-Motor43-GetUp-Recovery-v0', 'bipedal_locomotion_motor43.recovery.getup_env_cfg:WFGetUpRecoveryEnvCfg', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AGetUpRecoveryPPORunnerCfg'),
    ('Isaac-Motor43-GetUp-Recovery-Play-v0', 'bipedal_locomotion_motor43.recovery.getup_env_cfg:WFGetUpRecoveryEnvCfg_PLAY', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AGetUpRecoveryPPORunnerCfg'),
    ('Isaac-Motor43-GetUp-Auto-v0', 'bipedal_locomotion_motor43.recovery.getup_env_cfg:WFGetUpAutoEnvCfg', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AGetUpAutoPPORunnerCfg'),
    ('Isaac-Motor43-GetUp-Auto-Play-v0', 'bipedal_locomotion_motor43.recovery.getup_env_cfg:WFGetUpAutoEnvCfg_PLAY', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AGetUpAutoPPORunnerCfg'),
    ('Isaac-Motor43-GetUp-Bounded-v0', 'bipedal_locomotion_motor43.recovery.getup_env_cfg:WFGetUpAutoEnvCfg', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AGetUpBoundedPPORunnerCfg'),
    ('Isaac-Motor43-GetUp-Bounded-Play-v0', 'bipedal_locomotion_motor43.recovery.getup_env_cfg:WFGetUpAutoEnvCfg_PLAY', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AGetUpBoundedPPORunnerCfg'),
    ('Isaac-Motor43-GetUp-Inverted-v0', 'bipedal_locomotion_motor43.recovery.getup_env_cfg:WFInvertedGetUpEnvCfg', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AInvertedPPORunnerCfg'),
    ('Isaac-Motor43-GetUp-Inverted-Play-v0', 'bipedal_locomotion_motor43.recovery.getup_env_cfg:WFInvertedGetUpEnvCfg_PLAY', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AInvertedPPORunnerCfg'),
    ('Isaac-Motor43-Recovery-Progressive-v0', 'bipedal_locomotion_motor43.recovery.progressive_env_cfg:WFProgressiveRecoveryEnvCfg', 'bipedal_locomotion_motor43.recovery.agents:WFProgressiveRecoveryPPORunnerCfg'),
    ('Isaac-Motor43-Recovery-Progressive-Play-v0', 'bipedal_locomotion_motor43.recovery.progressive_env_cfg:WFProgressiveRecoveryEnvCfg_PLAY', 'bipedal_locomotion_motor43.recovery.agents:WFProgressiveRecoveryPPORunnerCfg'),
    ('Isaac-Motor43-Recovery-Inverted-Play-v0', 'bipedal_locomotion_motor43.recovery.progressive_env_cfg:WFInvertedRecoveryEnvCfg_PLAY', 'bipedal_locomotion_motor43.recovery.agents:WFProgressiveRecoveryPPORunnerCfg'),
    ('Isaac-Motor43-FallenPose-Generate-v0', 'bipedal_locomotion_motor43.recovery.fallen_pose_env_cfg:WFFallenPoseGenerationEnvCfg', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AFallenPPORunnerCfg'),
    ('Isaac-Motor43-Recovery-Fallen-v0', 'bipedal_locomotion_motor43.recovery.fallen_pose_env_cfg:WFFallenRecoveryEnvCfg', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AFallenPPORunnerCfg'),
    ('Isaac-Motor43-Recovery-Fallen-Play-v0', 'bipedal_locomotion_motor43.recovery.fallen_pose_env_cfg:WFFallenRecoveryEnvCfg_PLAY', 'bipedal_locomotion_motor43.recovery.agents:WF_TRON1AFallenPPORunnerCfg'),
):
    gym.register(id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": config, "rsl_rl_cfg_entry_point": runner})
