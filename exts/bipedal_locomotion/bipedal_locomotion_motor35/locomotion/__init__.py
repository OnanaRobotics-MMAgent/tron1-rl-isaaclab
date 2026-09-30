"""Explicit task registrations for this robot; no legacy aliases."""
import gymnasium as gym

for name, config, runner in (
    ('Isaac-Motor35-Locomotion-v0', 'bipedal_locomotion_motor35.locomotion.env_cfg:Motor35LocomotionEnvCfg', 'bipedal_locomotion_motor35.locomotion.agents:Motor35LocomotionRunnerCfg'),
    ('Isaac-Motor35-Locomotion-Play-v0', 'bipedal_locomotion_motor35.locomotion.env_cfg:Motor35LocomotionPlayEnvCfg', 'bipedal_locomotion_motor35.locomotion.agents:Motor35LocomotionRunnerCfg'),
):
    gym.register(id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": config, "rsl_rl_cfg_entry_point": runner})
