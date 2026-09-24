"""Headless Motor35 jump-reference scan, without a trained policy."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=110)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.tasks.locomotion.mdp.motor35_jump import jump_measurements
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg("Isaac-Motor35-Jump-Small-Play-v0", num_envs=args.num_envs, device=args.device)
    env = gym.make("Isaac-Motor35-Jump-Small-Play-v0", cfg=cfg)
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
    variants = (torch.linspace(-0.5, 0.5, args.num_envs, device=core.device)
                if args.num_envs > 1 else torch.zeros(1, device=core.device))
    try:
        for step in range(args.max_steps):
            actions = torch.zeros((args.num_envs, core.action_manager.total_action_dim), device=core.device)
            actions[:, 2] = variants
            actions[:, 3] = -variants
            actions[:, 4] = variants
            actions[:, 5] = -variants
            _, _, terminated, truncated, _ = env.step(actions)
            clearance, airborne, contact = jump_measurements(
                core, wheel, sensor, 0.04)
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
                print(f"step={step + 1} root_z={robot.data.root_pos_w[0, 2].item():.3f} "
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
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
