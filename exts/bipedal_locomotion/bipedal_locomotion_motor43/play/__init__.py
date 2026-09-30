"""Explicit task registrations for this robot; no legacy aliases."""
import gymnasium as gym

for name, config, runner in (
    ('Isaac-Motor43-Recovery-Locomotion-Play-v0', 'bipedal_locomotion_motor43.play.env_cfg:WFRecoveryLocomotionEnvCfg_PLAY', 'bipedal_locomotion_motor43.recovery.agents:WFProgressiveRecoveryPPORunnerCfg'),
):
    gym.register(id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": config, "rsl_rl_cfg_entry_point": runner})
