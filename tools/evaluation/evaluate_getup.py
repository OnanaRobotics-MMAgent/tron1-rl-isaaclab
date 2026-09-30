"""Evaluate fixed GetUp stages and audit executable action bounds.

Each environment contributes the same number of episodes. No PPO update is run.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Isaac-Motor43-GetUp-Auto-Play-v0')
parser.add_argument('--checkpoint_paths', nargs='+', required=True)
parser.add_argument('--stages', type=int, nargs='+', default=[1, 2])
parser.add_argument('--num_envs', type=int, default=128)
parser.add_argument('--episodes_per_env', type=int, default=4)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes_per_env < 1:
    parser.error('--episodes_per_env must be positive')
app = AppLauncher(args).app

import json
from pathlib import Path
import torch
import gymnasium as gym
import bipedal_locomotion_motor43, bipedal_locomotion_motor35  # task registration
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from bipedal_locomotion_common.wrappers.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runner import OnPolicyRunner

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.seed = args.seed
cfg.getup.curriculum_enabled = False
cfg.observations.policy.enable_corruption = args.stochastic
cfg.observations.obsHistory.enable_corruption = args.stochastic
for stage in args.stages:
    if not 0 <= stage < len(cfg.getup.tilt_ranges_deg):
        raise ValueError(f'Invalid stage {stage}')
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
raw = env.unwrapped
agent_cfg = load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point')
runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=raw.device)
bounds = torch.tensor(env.get_action_mean_bounds(), device=raw.device)
leg_ids, leg_names = raw.scene['robot'].find_joints('(abad|hip|knee)_[LR]_Joint')
results = []
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)

for checkpoint in args.checkpoint_paths:
    runner.load(checkpoint)
    policy = runner.get_inference_policy()
    encoder = runner.get_inference_encoder()
    for stage in args.stages:
        raw.task_state.level = stage
        # Each fixed-stage evaluation starts with independent episode counters.
        raw.task_state.active.zero_()
        raw.task_state.window_episodes = 0
        raw.task_state.window_successes = 0
        raw.task_state.direction_episodes = [0] * 4
        raw.task_state.direction_successes = [0] * 4
        raw.task_state.metric_windows.clear()
        raw.seed(args.seed)
        completed = torch.zeros(args.num_envs, dtype=torch.long, device=raw.device)
        length = torch.zeros_like(completed)
        records = []
        excess_sum = torch.zeros(env.num_actions, device=raw.device)
        clip_sum = torch.zeros_like(excess_sum)
        action_samples = 0
        with torch.inference_mode():
            obs, info = env.reset()
            direction = raw.task_state.direction.clone()
            gravity = raw.scene['robot'].data.projected_gravity_b
            initial_tilt = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1)))
            qmin = torch.full((len(leg_ids),), float('inf'), device=raw.device)
            qmax = -qmin
            while not bool((completed >= args.episodes_per_env).all()):
                data = raw.scene['robot'].data
                previous_h = (data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]).clone()
                previous_tilt = torch.rad2deg(torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1)))
                previous_q = data.joint_pos[:, leg_ids].clone()
                qmin = torch.minimum(qmin, previous_q.amin(0))
                qmax = torch.maximum(qmax, previous_q.amax(0))
                x = torch.cat((encoder(info['observations']['obsHistory'].flatten(1)), obs,
                               info['observations']['commands']), dim=-1)
                act = runner.alg.actor_critic.act(x) if args.stochastic else policy(x)
                active = completed < args.episodes_per_env
                excess = (bounds[:, 0] - act).clamp_min(0) + (act - bounds[:, 1]).clamp_min(0)
                excess_sum += excess[active].sum(0)
                clip_sum += (excess[active] > 1e-5).sum(0)
                action_samples += int(active.sum())
                obs, _, done, info = env.step(act)
                length += 1
                accepted = done.bool() & active
                success = raw.termination_manager.get_term('success')
                violation = raw.termination_manager.get_term('leg_limit_violation')
                for i in accepted.nonzero().flatten().tolist():
                    records.append(dict(success=bool(success[i]), direction=int(direction[i]),
                                        tilt=float(initial_tilt[i]), seconds=float(length[i] * raw.step_dt),
                                        previous_height=float(previous_h[i]), previous_tilt=float(previous_tilt[i]),
                                        previous_joint_pos=previous_q[i].tolist(), action=act[i].tolist(),
                                        leg_limit_violation=bool(violation[i])))
                completed += accepted.long()
                reset = done.bool()
                length[reset] = 0
                direction[reset] = raw.task_state.direction[reset]
                gravity = raw.scene['robot'].data.projected_gravity_b
                initial_tilt[reset] = torch.rad2deg(torch.acos((-gravity[reset, 2]).clamp(-1, 1)))
        successes = [r for r in records if r['success']]
        direction_stats = []
        for d in range(4):
            subset = [r for r in records if r['direction'] == d]
            direction_stats.append(dict(direction=d, episodes=len(subset),
                                        success_rate=sum(r['success'] for r in subset) / len(subset) if subset else None))
        result = dict(checkpoint=checkpoint, stage=stage, tilt_range=cfg.getup.tilt_ranges_deg[stage],
                      stochastic=args.stochastic, seed=args.seed, episodes=len(records),
                      success_rate=len(successes) / len(records),
                      mean_success_time=sum(r['seconds'] for r in successes) / len(successes) if successes else None,
                      direction_stats=direction_stats,
                      mean_action_excess=(excess_sum / action_samples).tolist(),
                      clipped_action_fraction=(clip_sum / action_samples).tolist(),
                      executable_action_bounds=bounds.tolist(),
                      leg_limit_violations=sum(r['leg_limit_violation'] for r in records),
                      joint_range_degrees={name: [float(torch.rad2deg(qmin[i])), float(torch.rad2deg(qmax[i]))]
                                           for i, name in enumerate(leg_names)}, records=records)
        results.append(result)
        output.write_text(json.dumps(results, indent=2))
        print('EVALUATION', json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
env.close()
app.close()
