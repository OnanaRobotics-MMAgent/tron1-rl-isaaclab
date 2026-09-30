"""Explicit task registrations for this robot; no legacy aliases."""
import gymnasium as gym

for name, config, runner in (
    ('Isaac-Motor35-Recovery-Progressive-v0', 'bipedal_locomotion_motor35.recovery.env_cfg:Motor35RecoveryEnvCfg', 'bipedal_locomotion_motor35.recovery.agents:Motor35RecoveryRunnerCfg'),
    ('Isaac-Motor35-Recovery-Inverted-Play-v0', 'bipedal_locomotion_motor35.recovery.env_cfg:Motor35RecoveryPlayEnvCfg', 'bipedal_locomotion_motor35.recovery.agents:Motor35RecoveryRunnerCfg'),
):
    gym.register(id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": config, "rsl_rl_cfg_entry_point": runner})
