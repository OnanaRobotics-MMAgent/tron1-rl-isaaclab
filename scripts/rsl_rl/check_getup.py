"""Simulator integration checks for resets, observation contracts and finite rollouts."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", default="Isaac-Limx-WF-GetUp-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.tasks.recovery.mdp.events import reset_fallen
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.episode_length_s = 1.0  # exercise automatic reset / timeout for every stage
    env = gym.make(args.task, cfg=cfg)
    try:
        raw = env.unwrapped
        for level in range(len(cfg.getup.tilt_ranges_deg)):
            raw.task_state.level = level
            obs, _ = env.reset()
            if cfg.getup.require_landing:
                up_z = -raw.scene["robot"].data.projected_gravity_b[:, 2]
                assert (up_z <= -torch.cos(torch.deg2rad(torch.tensor(10.0)))).all(), up_z
                assert not raw.task_state.control_ready.any()
            assert obs["policy"].shape == (args.num_envs, 28)
            assert obs["obsHistory"].shape == (args.num_envs, 10, 28)
            assert torch.count_nonzero(obs["commands"]) == 0
            asset = raw.scene["robot"]
            leg_ids = raw._getup_leg_joint_ids
            limits = asset.data.joint_pos_limits[:, leg_ids]
            assert torch.isfinite(limits).all() and (limits.abs() < torch.pi).all()
            leg_action = raw.action_manager.get_term("joint_pos")
            leg_action.process_actions(torch.full((args.num_envs, 6), 1e6, device=raw.device))
            assert (leg_action.processed_actions <= asset.data.joint_pos_limits[:, leg_action._joint_ids, 1]).all()
            leg_action.process_actions(torch.full((args.num_envs, 6), -1e6, device=raw.device))
            assert (leg_action.processed_actions >= asset.data.joint_pos_limits[:, leg_action._joint_ids, 0]).all()
            root = asset.data.root_state_w.clone()
            corners = raw._getup_collision_corners
            rotation = root[:, None, 3:7].expand(-1, len(corners), -1).reshape(-1, 4)
            points = corners[None].expand(args.num_envs, -1, -1).reshape(-1, 3)
            minimum = quat_apply(rotation, points).reshape(args.num_envs, -1, 3)[:, :, 2].amin(dim=1)
            minimum += root[:, 2] - raw.scene.env_origins[:, 2]
            assert (minimum >= cfg.getup.reset_clearance - 1e-5).all(), minimum
            # A subset reset must leave every other physical state untouched.
            reset_fallen(raw, torch.tensor([0], device=raw.device))
            torch.testing.assert_close(asset.data.root_state_w[1:], root[1:])
            assert (raw.task_state.stage == level).all()
            assert not raw.task_state.success.any()
            terminations = 0
            ever_ready = torch.zeros(args.num_envs, device=raw.device, dtype=torch.bool)
            ever_touch = torch.zeros_like(ever_ready)
            for _ in range(60):
                actions = torch.randn(args.num_envs, 8, device=raw.device) * 0.15
                obs, reward, terminated, truncated, _ = env.step(actions)
                if cfg.getup.require_landing:
                    ever_ready |= raw.task_state.control_ready
                    ever_touch |= raw.scene["contact_forces"].data.net_forces_w.norm(dim=-1).amax(dim=1) > cfg.getup.landing_force
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                terminations += int((terminated | truncated).sum())
            assert terminations >= args.num_envs, "Timeout / automatic reset did not run."
            if cfg.getup.require_landing:
                print(f"[GetUp check] touched={int(ever_touch.sum())}/{args.num_envs}, "
                      f"control_released={int(ever_ready.sum())}/{args.num_envs}")
                assert ever_ready.all(), "Some robots never gained control after ground contact."
            print(f"[GetUp check] stage={level}: collision clearance, subset reset, observations, rollout passed")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
