"""Play fixed inverted recovery and save physics torques, means and peaks for every episode."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'rsl_rl'), str(ROOT / 'exts/bipedal_locomotion')]
import rsl_rl  # pin the encoder-enabled package before Kit startup
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint_path', required=True)
parser.add_argument('--episodes', type=int, default=20, help='Number of complete recovery episodes (one robot).')
parser.add_argument('--seed', type=int, default=20260924)
parser.add_argument('--output_dir', help='New directory; existing directories are never overwritten.')
parser.add_argument('--real_time', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
checkpoint = Path(args.checkpoint_path).expanduser().resolve()
if not checkpoint.is_file() or args.episodes < 1:
    parser.error('Provide an existing checkpoint and a positive episode count')
output = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
    ROOT / 'logs/recovery_torques' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f'))
if output.exists():
    parser.error(f'Output directory already exists: {output}')
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import bipedal_locomotion
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runner import OnPolicyRunner
from torque_report import TorqueReport


def main():
    task = 'Isaac-Limx-WF-Recovery-Inverted-Play-v0'
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
    raw = env.unwrapped
    robot = raw.scene['robot']
    report = TorqueReport(output, list(robot.joint_names), float(raw.physics_dt), cfg.decimation,
                         dict(task=task, checkpoint=str(checkpoint), seed=args.seed,
                              checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              tilt_range_deg=[170, 180], requested_episodes=args.episodes,
                              self_collisions=True, deterministic_policy=True))
    manager = raw.recorder_manager
    original = manager.record_post_physics_decimation_step
    samples = []
    initial = {}

    def sample():
        original()
        # Called after EACH physics solve, before automatic episode reset.
        pd = robot.data.applied_torque[0].detach().cpu().numpy().copy()
        measured = robot.root_physx_view.get_dof_projected_joint_forces()[0].detach().cpu().numpy().copy()
        samples.append(np.stack((pd, measured)))

    def initial_state():
        tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[0, 2]).clamp(-1, 1)))
        return dict(initial_tilt_deg=float(tilt), direction=int(raw.task_state.direction[0]),
                    initial_joint_positions_rad=robot.data.joint_pos[0].cpu().tolist())

    try:
        agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
        runner = OnPolicyRunner(env, agent.to_dict(), device=raw.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy()
        encoder = runner.get_inference_encoder()
        obs, info = env.reset()
        initial = initial_state()
        manager.record_post_physics_decimation_step = sample
        print(f'[RecoveryTorques] Writing {args.episodes} episodes to {output}', flush=True)
        while len(report.episodes) < args.episodes and app.is_running():
            started = time.monotonic()
            with torch.inference_mode():
                inputs = torch.cat((encoder(info['observations']['obsHistory'].flatten(1)),
                                    obs, info['observations']['commands']), dim=-1)
                obs, _, dones, info = env.step(policy(inputs))
            if bool(dones[0]):
                reasons = [name for name in raw.termination_manager.active_terms
                           if bool(raw.termination_manager.get_term(name)[0])]
                # Snapshot before disk/plot operations: next episode has already reset.
                saved, samples = samples, []
                report.save_episode(saved, dict(initial, complete=True, outcome='+'.join(reasons)))
                print(f'[RecoveryTorques] {len(report.episodes)}/{args.episodes}: {reasons}', flush=True)
                initial = initial_state()
            if args.real_time or not args.headless:
                time.sleep(max(0., raw.step_dt - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print('[RecoveryTorques] Interrupted; saving partial episode.', flush=True)
    finally:
        manager.record_post_physics_decimation_step = original
        try:
            if samples:
                report.save_episode(samples, dict(initial, complete=False, outcome='interrupted_or_error'))
            report.finish()
            print(f'[RecoveryTorques] Results: {output}', flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
