"""Explicit task registrations for this robot; no legacy aliases."""
import gymnasium as gym

for name, config, runner in (
    ('Isaac-Motor35-Recovery-Locomotion-Play-v0', 'bipedal_locomotion_motor35.play.env_cfg:Motor35RecoveryLocomotionPlayEnvCfg', 'bipedal_locomotion_motor35.recovery.agents:Motor35RecoveryRunnerCfg'),
):
    gym.register(id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": config, "rsl_rl_cfg_entry_point": runner})
