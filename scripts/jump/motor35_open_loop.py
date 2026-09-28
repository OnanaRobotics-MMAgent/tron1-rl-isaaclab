"""Headless Motor35 jump-reference scan, without a trained policy."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=110)
parser.add_argument("--task", default="Isaac-Motor35-Jump-Small-Play-v0")
parser.add_argument("--residual_span", type=float, default=0.0)
parser.add_argument("--trigger_delay", type=float, default=0.7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.tasks.locomotion.mdp.motor35_jump import jump_measurements, get_state
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg(args.task, num_envs=args.num_envs, device=args.device)
    cfg.jump.jump_probability = 1.0
    cfg.jump.trigger_delay_range = (args.trigger_delay, args.trigger_delay)
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    core = env.unwrapped
    wheel = SceneEntityCfg("robot", body_names=["wheel_L_Link", "wheel_R_Link"])
    sensor = SceneEntityCfg("contact_forces", body_names=["wheel_L_Link", "wheel_R_Link"])
    wheel.resolve(core.scene)
    sensor.resolve(core.scene)
    peaks = torch.zeros(args.num_envs, device=core.device)
    raw_peaks = torch.full_like(peaks, -1.0)
    took_off = torch.zeros(args.num_envs, dtype=torch.bool, device=core.device)
    landed = torch.zeros_like(took_off)
    resets = torch.zeros(args.num_envs, dtype=torch.long, device=core.device)
    variants = (torch.linspace(-args.residual_span, args.residual_span, args.num_envs, device=core.device)
                if args.num_envs > 1 else torch.zeros(1, device=core.device))
    hip_ids = core.scene["robot"].find_bodies(["hip_L_Link", "hip_R_Link"], preserve_order=True)[0]
    fk_error_max = 0.0
    torque_max = 0.0
    phase_counts = torch.zeros(6, device=core.device)
    try:
        for step in range(args.max_steps):
            actions = torch.zeros((args.num_envs, core.action_manager.total_action_dim), device=core.device)
            actions[:, 2] = variants
            actions[:, 3] = -variants
            actions[:, 4] = variants
            actions[:, 5] = -variants
            _, _, terminated, truncated, _ = env.step(actions)
            state = get_state(core)
            robot_data = core.scene["robot"].data
            torque_max = max(torque_max, robot_data.applied_torque.abs().max().item())
            # Hip link origin is the hip joint anchor. Remove fixed lateral
            # hip->wheel offset to compare sagittal FK against the USD chain.
            delta = robot_data.body_pos_w[:, wheel.body_ids] - robot_data.body_pos_w[:, hip_ids]
            measured = (delta.square().sum(-1) - .0251**2).clamp_min(0).sqrt()
            valid = ~(terminated | truncated)
            if valid.any():
                fk_error_max = max(fk_error_max, (measured[valid]-state.length[valid]).abs().max().item())
            phase_counts += torch.bincount(state.phase, minlength=6)
            clearance, airborne, contact = jump_measurements(
                core, wheel, sensor, 0.04)
            # Exclude reset spawn clearance and falling-over wheel lifts.
            airborne &= state.had_flight & (state.phase == 3) & valid
            raw_peaks = torch.maximum(raw_peaks, clearance)
            peaks = torch.where(airborne, torch.maximum(peaks, clearance), peaks)
            landed |= took_off & contact
            took_off |= airborne
            resets += (terminated | truncated).long()
            if step in (9, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 40, 50, 60, 70, 80):
                robot = core.scene["robot"]
                action_term = core.action_manager.get_term("joint_pos")
                actual = robot.data.joint_pos[0, action_term._joint_ids]
                target = action_term.processed_actions[0]
                forces = core.scene["contact_forces"].data.net_forces_w[0, sensor.body_ids].norm(dim=-1)
                base_ids, _ = robot.find_bodies("base_Link")
                base_force = core.scene["contact_forces"].data.net_forces_w[0, base_ids].norm().item()
                print(f"step={step + 1} phase={state.phase[0].item()} root_z={robot.data.root_pos_w[0, 2].item():.3f} "
                      f"root_xy={[round(x, 3) for x in robot.data.root_pos_w[0, :2].tolist()]} "
                      f"tilt={robot.data.projected_gravity_b[0, :2].norm().item():.3f} "
                      f"min_clearance={clearance[0].item():.3f} "
                      f"base_N={base_force:.1f} reset={bool((terminated | truncated)[0])} "
                      f"contacts_N={[round(x, 1) for x in forces.tolist()]} "
                      f"hip_knee_target={[round(x, 2) for x in target[2:].tolist()]} "
                      f"hip_knee_actual={[round(x, 2) for x in actual[2:].tolist()]}")
        for idx in range(args.num_envs):
            print(f"env={idx} residual={variants[idx].item():+.2f} "
                  f"takeoff={bool(took_off[idx])} peak_clearance_m={peaks[idx].item():.4f} "
                  f"max_raw_clearance_m={raw_peaks[idx].item():.4f} "
                  f"landed={bool(landed[idx])} resets={int(resets[idx])}")
        print(f"phase_counts={phase_counts.tolist()} fk_error_max_m={fk_error_max:.6f} torque_max_nm={torque_max:.4f}")
        print("jump_metrics", {key: float(value) for key, value in get_state(core).metrics(slice(None)).items()})
        assert fk_error_max < .002, f"URDF/USD FK mismatch: {fk_error_max}"
        assert torque_max <= 3.001, f"Motor torque limit exceeded: {torque_max}"
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
