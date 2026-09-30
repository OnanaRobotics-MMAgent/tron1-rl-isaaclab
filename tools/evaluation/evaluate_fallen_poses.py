"""Evaluate each held-out settled pose once, without fast-episode sampling bias."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'exts/bipedal_locomotion'), str(ROOT / 'rsl_rl')]
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', required=True)
p.add_argument('--pose_bank', required=True)
p.add_argument('--output', required=True)
p.add_argument('--num_envs', type=int, default=128)
p.add_argument('--seed', type=int, default=8128)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import json
import torch
import gymnasium as gym
import bipedal_locomotion_motor43, bipedal_locomotion_motor35
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from bipedal_locomotion_common.wrappers.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runner import OnPolicyRunner

task = 'Isaac-Motor43-Recovery-Fallen-Play-v0'
cfg = parse_env_cfg(task, device=a.device, num_envs=a.num_envs)
cfg.fallen.bank_path = a.pose_bank
cfg.seed = a.seed
env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
raw = env.unwrapped
runner = OnPolicyRunner(env, load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point').to_dict(), device=raw.device)
runner.load(a.checkpoint)
policy = runner.get_inference_policy(); encoder = runner.get_inference_encoder()
records = []
count = len(raw._fallen_bank['root_state'])
try:
    with torch.inference_mode():
        for start in range(0, count, a.num_envs):
            size = min(a.num_envs, count-start)
            ids = torch.arange(start, start+a.num_envs, device=raw.device).clamp_max(count-1)
            raw._fallen_forced_indices = ids
            obs, info = env.reset()
            raw._fallen_forced_indices = None
            # Verify the policy really starts from the random-joint snapshot,
            # rather than accidentally evaluating the previous neutral reset.
            torch.testing.assert_close(raw.scene['robot'].data.joint_pos,
                                       raw._fallen_bank['joint_pos'][ids], atol=1e-3, rtol=1e-3)
            assert bool((torch.acos((-raw.scene['robot'].data.projected_gravity_b[:, 2]).clamp(-1, 1)) > .9).all())
            completed = torch.arange(a.num_envs, device=raw.device) >= size
            elapsed = torch.zeros(a.num_envs, dtype=torch.long, device=raw.device)
            for _ in range(raw.max_episode_length+1):
                x = torch.cat((encoder(info['observations']['obsHistory'].flatten(1)), obs, info['observations']['commands']), -1)
                obs, _, done, info = env.step(policy(x))
                elapsed += 1
                accepted = done.bool() & ~completed
                for i in accepted.nonzero().flatten().tolist():
                    idx = int(ids[i]);data = raw._fallen_bank
                    records.append(dict(pose_id=idx, stage=int(data['stage'][idx]), direction=int(data['direction'][idx]),
                                        success=bool(raw.termination_manager.get_term('success')[i]),
                                        leg_limit_violation=bool(raw.termination_manager.get_term('leg_limit_violation')[i]),
                                        timeout=bool(raw.termination_manager.get_term('time_out')[i]),
                                        out_of_bounds=bool(raw.termination_manager.get_term('out_of_bounds')[i]),
                                        seconds=float(elapsed[i]*raw.step_dt)))
                completed |= done.bool()
                if bool(completed.all()):break
            if not bool(completed.all()):raise RuntimeError('Evaluation failed to complete all episodes')
            print(f'Evaluated {len(records)}/{count}', flush=True)
    assert len(records) == count and len({r['pose_id'] for r in records}) == count
    stages = []
    for s in range(3):
        subset = [r for r in records if r['stage']==s]
        directions = []
        for d in range(4):
            group=[r for r in subset if r['direction']==d]
            directions.append(dict(direction=d, episodes=len(group),success_rate=sum(r['success'] for r in group)/len(group) if group else None))
        stages.append(dict(stage=s,episodes=len(subset), success_rate=sum(r['success'] for r in subset)/len(subset) if subset else None,
                           direction_stats=directions, leg_limit_violations=sum(r['leg_limit_violation'] for r in subset)))
    result=dict(checkpoint=str(Path(a.checkpoint).resolve()),pose_bank=str(Path(a.pose_bank).resolve()),split='test',seed=a.seed,
                episodes=count,success_rate=sum(r['success'] for r in records)/count,
                initial_state_verified=True,
                leg_limit_violations=sum(r['leg_limit_violation'] for r in records),stages=stages,records=records)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(output)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
finally:
    env.close();app.close()
