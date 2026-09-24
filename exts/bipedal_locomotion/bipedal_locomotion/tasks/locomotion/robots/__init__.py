import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import PF_TRON1AFlatPPORunnerCfg, WF_TRON1AFlatPPORunnerCfg, SF_TRON1AFlatPPORunnerCfg

from . import limx_pointfoot_env_cfg, limx_wheelfoot_env_cfg, limx_solefoot_env_cfg, motor35_wheelfoot_env_cfg, motor35_jump_env_cfg

##
# Create PPO runners for RSL-RL
##

limx_pf_blind_flat_runner_cfg = PF_TRON1AFlatPPORunnerCfg()

limx_wf_blind_flat_runner_cfg = WF_TRON1AFlatPPORunnerCfg()

motor35_wf_blind_flat_runner_cfg = WF_TRON1AFlatPPORunnerCfg().replace(experiment_name="motor35_wf_flat")
motor35_jump_small_runner_cfg = WF_TRON1AFlatPPORunnerCfg().replace(
    experiment_name="motor35_jump_small", max_iterations=4000
)
motor35_jump_high_runner_cfg = WF_TRON1AFlatPPORunnerCfg().replace(
    experiment_name="motor35_jump_high", max_iterations=10000
)

limx_sf_blind_flat_runner_cfg = SF_TRON1AFlatPPORunnerCfg()



##
# Register Gym environments
##

############################
# PF Blind Flat Environment
############################
gym.register(
    id="Isaac-Limx-PF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-PF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

#############################
# WF Blind Flat Environment
#############################
gym.register(
    id="Isaac-Limx-WF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Motor35-WF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motor35_wheelfoot_env_cfg.Motor35WFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": motor35_wf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Motor35-WF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motor35_wheelfoot_env_cfg.Motor35WFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": motor35_wf_blind_flat_runner_cfg,
    },
)

for task_id, config, runner in (
    ("Isaac-Motor35-Jump-Small-v0", motor35_jump_env_cfg.Motor35JumpSmallEnvCfg,
     motor35_jump_small_runner_cfg),
    ("Isaac-Motor35-Jump-Small-Play-v0", motor35_jump_env_cfg.Motor35JumpSmallEnvCfg_PLAY,
     motor35_jump_small_runner_cfg),
    ("Isaac-Motor35-Jump-High-v0", motor35_jump_env_cfg.Motor35JumpHighEnvCfg,
     motor35_jump_high_runner_cfg),
    ("Isaac-Motor35-Jump-High-Play-v0", motor35_jump_env_cfg.Motor35JumpHighEnvCfg_PLAY,
     motor35_jump_high_runner_cfg),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": config, "rsl_rl_cfg_entry_point": runner},
    )


############################
# SF Blind Flat Environment
############################
gym.register(
    id="Isaac-Limx-SF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)
