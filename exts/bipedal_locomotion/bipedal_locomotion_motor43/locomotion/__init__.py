"""Explicit task registrations for this robot; no legacy aliases."""
import gymnasium as gym

for name, config, runner in (
    ('Isaac-Motor43-Locomotion-v0', 'bipedal_locomotion_motor43.locomotion.env_cfg:WFBlindFlatEnvCfg', 'bipedal_locomotion_motor43.locomotion.agents:WF_TRON1AFlatPPORunnerCfg'),
    ('Isaac-Motor43-Locomotion-Play-v0', 'bipedal_locomotion_motor43.locomotion.env_cfg:WFBlindFlatEnvCfg_PLAY', 'bipedal_locomotion_motor43.locomotion.agents:WF_TRON1AFlatPPORunnerCfg'),
):
    gym.register(id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": config, "rsl_rl_cfg_entry_point": runner})
