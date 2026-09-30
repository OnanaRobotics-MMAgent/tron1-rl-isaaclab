"""Inverted recovery -> encoder/IMU geometry gate -> smoothly blended locomotion."""
import argparse
from datetime import datetime
import json
import csv
import importlib
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'rsl_rl'), str(ROOT / 'exts/bipedal_locomotion')]
import rsl_rl
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--robot', choices=('motor43', 'motor35'), default='motor43')
parser.add_argument('--recovery_checkpoint')
parser.add_argument('--locomotion_checkpoint')
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--duration', type=float, default=30.)
parser.add_argument('--forward_speed', type=float, default=.3, help='Rolling speed in m/s: Motor43 body-y, Motor35 body-x.')
parser.add_argument('--yaw_rate', type=float, default=0., help='Yaw rate in rad/s.')
parser.add_argument('--blend_time', type=float, default=1.)
parser.add_argument('--ready_hold', type=float, default=.3)
parser.add_argument('--seed', type=int, default=20260924)
parser.add_argument('--output_dir')
parser.add_argument('--video', action='store_true')
parser.add_argument('--video_folder', help='Recording directory; defaults to videos/<robot>/<timestamp>.')
parser.add_argument('--real_time', action='store_true')
parser.add_argument('--current_asset', action='store_true', help='Use current branch model/default pose instead of the supplied checkpoints training asset.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
profile = importlib.import_module(f'bipedal_locomotion_{args.robot}.play.profile')
if args.robot == 'motor43':
    args.recovery_checkpoint = args.recovery_checkpoint or str(ROOT / 'checkpoints/motor43/recovery/model_30000.pt')
    args.locomotion_checkpoint = args.locomotion_checkpoint or str(ROOT / 'checkpoints/motor43/locomotion/model_10000.pt')
if not args.recovery_checkpoint or not args.locomotion_checkpoint:
    parser.error('Provide both --recovery_checkpoint and --locomotion_checkpoint for Motor35')
import math
if (args.num_envs < 1 or not all(math.isfinite(x) for x in (args.duration, args.blend_time, args.ready_hold, args.forward_speed, args.yaw_rate))
        or min(args.duration, args.blend_time, args.ready_hold) <= 0
        or abs(args.forward_speed) > profile.COMMAND_LIMIT or abs(args.yaw_rate) > profile.YAW_LIMIT):
    parser.error('Invalid timing/env count, or velocity exceeds trained command ranges')
for path in (args.recovery_checkpoint, args.locomotion_checkpoint):
    if not Path(path).is_file():
        parser.error(f'Checkpoint missing: {path}')
output = Path(args.output_dir) if args.output_dir else ROOT / 'outputs/recovery_handoff' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
output = output.resolve()
output.mkdir(parents=True, exist_ok=False)
video_folder = (Path(args.video_folder).resolve() if args.video_folder else
                ROOT / 'videos' / args.robot / datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f'))
args.enable_cameras = args.video
app = AppLauncher(args).app

import torch
import gymnasium as gym
import bipedal_locomotion_motor43, bipedal_locomotion_motor35
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from bipedal_locomotion_common.wrappers.rsl_rl import RslRlVecEnvWrapper
from bipedal_locomotion_common.play.handoff import WheelLegFK, HandoffController, handoff_conditions, blend_actions, CheckpointPolicy


def main():
    task = profile.TASK_ID
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.num_envs <= 64:
        # Play scenes do not need contact buffers sized for thousands of robots.
        cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
        cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    urdf, asset_revision = profile.configure_asset(cfg, args.current_asset)
    print(f'[Handoff] {args.robot} asset: {asset_revision}; {urdf}', flush=True)
    cfg.episode_length_s = max(args.duration + 1., 60.)
    cfg.viewer.resolution = (960, 720)
    cfg.viewer.eye = (0.8, 0.8, 0.5)
    cfg.viewer.lookat = (0., 0., 0.15)
    env = gym.make(task, cfg=cfg, render_mode='rgb_array' if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(env, video_folder=str(video_folder),
              step_trigger=lambda step: step == 0, video_length=math.ceil(args.duration/(cfg.sim.dt*cfg.decimation)),
              name_prefix='recovery_to_locomotion', disable_logger=True)
    env = RslRlVecEnvWrapper(env)
    raw, dt = env.unwrapped, env.unwrapped.step_dt
    robot = raw.scene['robot']
    trace = None
    transitions = []
    try:
        runners = []
        for path in (args.recovery_checkpoint, args.locomotion_checkpoint):
            runners.append(CheckpointPolicy(path, raw.device, getattr(cfg, 'policy_contract', None)))
        obs, info = env.reset()
        if obs.shape[1] != 28 or info['observations']['obsHistory'].flatten(1).shape[1] != 280:
            raise ValueError('Expected shared 28-dimensional observation and 10-frame history')
        fk = WheelLegFK(urdf, robot.joint_names, raw.device, wheel_axis=profile.WHEEL_AXIS)
        down = torch.tensor([0., 0., -1.], device=raw.device).expand(args.num_envs, -1)
        nominal = fk.compute(robot.data.default_joint_pos, down)['lengths']
        controller = HandoffController(args.num_envs, dt, raw.device, hold=args.ready_hold, blend=args.blend_time)
        wheel_ids, _ = robot.find_bodies(['wheel_L_Link', 'wheel_R_Link'], preserve_order=True)
        contact_ids, _ = raw.scene['contact_forces'].find_bodies(['wheel_L_Link', 'wheel_R_Link'], preserve_order=True)
        # Resolve actual action term order. Do not assume asset joint order matches it.
        pos, vel = raw.action_manager.get_term('joint_pos'), raw.action_manager.get_term('joint_vel')
        scales = torch.ones(args.num_envs, 8, device=raw.device)
        scales[:, :6], scales[:, 6:] = pos._scale, vel._scale
        offset = torch.zeros_like(scales)
        offset[:, :6], offset[:, 6:] = pos._offset, vel._offset
        if not torch.allclose(scales[0], torch.tensor(profile.ACTION_SCALES, device=raw.device)):
            raise ValueError('Checkpoint action scales do not match handoff environment')
        low = torch.cat((robot.data.joint_pos_limits[:, pos._joint_ids, 0], -profile.WHEEL_SPEED_LIMIT*torch.ones(args.num_envs, 2, device=raw.device)), -1)
        high = torch.cat((robot.data.joint_pos_limits[:, pos._joint_ids, 1], profile.WHEEL_SPEED_LIMIT*torch.ones(args.num_envs, 2, device=raw.device)), -1)
        rate = torch.tensor([3.]*6+[80.]*2, device=raw.device)
        previous = torch.zeros_like(scales)
        desired = torch.zeros(args.num_envs, 3, device=raw.device)
        desired[:, profile.FORWARD_AXIS], desired[:, 2] = args.forward_speed, args.yaw_rate
        dimensions = dict(policy_observation=28, history_frames=10, encoder_input=280, encoder_output=3,
                          command=3, actor_input=34, checkpoint_critic_inputs=[p.critic_input for p in runners], action=8)
        metadata = dict(task=task, checkpoints=[str(Path(p).resolve()) for p in (args.recovery_checkpoint, args.locomotion_checkpoint)],
                        checkpoint_sha256=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.recovery_checkpoint, args.locomotion_checkpoint)],
                        dimensions=dimensions, seed=args.seed, asset_revision=asset_revision,
                        urdf=str(urdf), robot=args.robot, forward_axis=profile.FORWARD_AXIS,
                        lateral_axis=profile.LATERAL_AXIS, separation_sign=profile.SEPARATION_SIGN,
                        self_collisions=True, blend_time_s=args.blend_time,
                        ready_hold_s=args.ready_hold, command=desired[0].tolist(),
                        thresholds=dict(tilt_deg=20, support_height_m=[.14,.23], height_difference_m=.035,
                                        length_ratio=[.7,1.35], downward_extension_m=.07, axle_tilt_deg=35,
                                        axle_alignment_deg=45, wheel_lateral_separation_m=.08, angular_speed=.8,
                                        wheel_force_N=3, retreat_tilt_deg=40, retreat_hold_s=.15),
                        note='FK plus IMU projected gravity; support height is geometric, not true world height. Contact is an additional gate. '
                             'Both policies retain their own encoder and share actual-action observation history. Absolute yaw is not inferred from gravity.')
        (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
        trace = (output/'trace.csv').open('w', newline='')
        writer = csv.writer(trace)
        writer.writerow(['time_s','env','phase','alpha','command_gain','ready','unsafe','tilt_deg',
                         'leg_length_L','leg_length_R','support_height_L','support_height_R','fk_error_m',
                         'body_forward_speed','body_wz',*[f'action_{i}' for i in range(8)],'reset',
                         'axle_vertical_max','axle_alignment','extension_min','lateral_separation','angular_speed','wheel_force_min'])
        max_fk_error = 0.
        print(f'[Handoff] Shared dimensions: {dimensions}; output={output}', flush=True)
        for step in range(math.ceil(args.duration/dt)):
            if not app.is_running():
                break
            started = time.monotonic()
            with torch.inference_mode():
                geom = fk.compute(robot.data.joint_pos, robot.data.projected_gravity_b)
                # Diagnostic ONLY: compare encoder-based FK with simulator link poses.
                centers_w = []
                for j, (_, center_offset, _, _) in enumerate(fk.wheels):
                    centers_w.append(robot.data.body_pos_w[:, wheel_ids[j]] +
                        quat_apply(robot.data.body_quat_w[:, wheel_ids[j]], center_offset.expand(args.num_envs, -1)))
                sim_centers = torch.stack([quat_apply_inverse(robot.data.root_quat_w, c-robot.data.root_pos_w) for c in centers_w], 1)
                fk_error = (geom['centers']-sim_centers).norm(dim=-1).amax(-1)
                max_fk_error = max(max_fk_error, float(fk_error.max()))
                if max_fk_error > .002:
                    raise RuntimeError(f'FK disagrees with asset by {max_fk_error} m')
                forces = raw.scene['contact_forces'].data.net_forces_w[:, contact_ids, 2]
                ready, unsafe = handoff_conditions(geom, nominal, robot.data.root_ang_vel_b, forces,
                    lateral_axis=profile.LATERAL_AXIS, separation_sign=profile.SEPARATION_SIGN)
                old = controller.phase.clone()
                alpha, command_gain = controller.update(ready, unsafe)
                for i in (old != controller.phase).nonzero().flatten().tolist():
                    event = dict(time_s=step*dt, env=i, previous=int(old[i]), phase=int(controller.phase[i]), tilt_deg=float(torch.rad2deg(geom['tilt'][i])))
                    transitions.append(event)
                    print('[Handoff]', event, flush=True)
                history = info['observations']['obsHistory'].flatten(1)
                commands = desired*command_gain[:, None]
                actions = []
                for k, runner in enumerate(runners):
                    cmd = torch.zeros_like(commands) if k == 0 else commands
                    actions.append(runner.act(obs, history, cmd))
                action = blend_actions(actions[0], actions[1], alpha, previous, scales, offset, low, high, rate, dt,
                                       (controller.phase != 0) | (old == 3))
                # Log pre-step state and the actual submitted action; dones belongs to this action's next state.
                metrics = torch.cat((controller.phase[:,None],alpha[:,None],command_gain[:,None],ready[:,None],unsafe[:,None],
                        torch.rad2deg(geom['tilt'])[:,None],geom['lengths'],geom['support_heights'],fk_error[:,None],
                        robot.data.root_lin_vel_b[:,profile.FORWARD_AXIS:profile.FORWARD_AXIS+1],robot.data.root_ang_vel_b[:,2:3],action),-1).cpu().tolist()
                diagnostics = torch.stack((geom['axle_vertical'].amax(-1),geom['axle_alignment'],geom['extensions'].amin(-1),
                    profile.SEPARATION_SIGN*(geom['centers'][:,1,profile.LATERAL_AXIS]-geom['centers'][:,0,profile.LATERAL_AXIS]),robot.data.root_ang_vel_b.norm(dim=-1),forces.amin(-1)),-1).cpu().tolist()
                obs, _, dones, info = env.step(action)
                if (args.video or not args.headless) and step % 2 == 0:
                    center = robot.data.root_pos_w[0].cpu().numpy()
                    raw.sim.set_camera_view(center + (0.8, 0.8, 0.5), center)
                previous = action.clone()
                for i, row in enumerate(metrics):
                    writer.writerow([step*dt, i, *row, int(dones[i]), *diagnostics[i]])
                if dones.any():
                    mask = dones.bool()
                    for i in mask.nonzero().flatten().tolist():
                        transitions.append(dict(time_s=(step+1)*dt, env=i, reset=True,
                            reasons=[name for name in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(name)[i])]))
                    controller.reset(mask)
                    previous[mask] = 0
            if args.real_time or not args.headless:
                time.sleep(max(0., dt-(time.monotonic()-started)))
        validation = dict(max_fk_error_m=max_fk_error,
                          entered_locomotion=len({e['env'] for e in transitions if e.get('phase') == 2}),
                          resets=sum(e.get('reset', False) for e in transitions),
                          retreats=sum(e.get('phase') == 3 for e in transitions))
        (output/'validation.json').write_text(json.dumps(validation, indent=2))
        if args.video:
            (video_folder/'playback.json').write_text(json.dumps(dict(metadata, validation=validation,
                duration_s=args.duration, transitions=transitions), indent=2)+'\n')
    finally:
        if trace:
            trace.close()
        (output/'transitions.json').write_text(json.dumps(transitions, indent=2))
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
