"""Replay an RSL-RL checkpoint and export per-joint torque curves."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

# Pin this repository's encoder-enabled RSL-RL package before Kit startup.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "rsl_rl"))
import rsl_rl  # noqa: F401

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Replay an RSL-RL checkpoint and plot applied joint torques.")
parser.add_argument("--task", type=str, default="Isaac-Limx-WF-Blind-Flat-Play-v0")
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=30.0, help="Requested simulation duration in seconds.")
parser.add_argument("--max_steps", type=int, default=None, help="Override duration with a policy-step count.")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--zero_actions", action="store_true", help="Smoke test without loading the policy.")
command_mode = parser.add_mutually_exclusive_group()
command_mode.add_argument("--stand_still", action="store_true", help="Replay the policy with a fixed zero base-velocity command.")
command_mode.add_argument("--max_speed", action="store_true", help="Fix body-y command to the task's positive maximum, with zero body-x/yaw command.")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1 or not math.isfinite(args_cli.duration) or args_cli.duration <= 0:
    parser.error("num_envs and duration must be positive and finite")
if args_cli.max_steps is not None and args_cli.max_steps < 1:
    parser.error("max_steps must be positive")
if not Path(args_cli.checkpoint_path).is_file():
    parser.error("checkpoint_path does not exist")
if args_cli.output_dir is None:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    args_cli.output_dir = str(Path(args_cli.checkpoint_path).resolve().parent / "torque_replay" / stamp)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
matplotlib.use("Agg" if args_cli.headless else "TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rsl_rl.runner import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, str(_REPO_ROOT / "exts/bipedal_locomotion"))
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, RslRlVecEnvWrapper
import bipedal_locomotion  # noqa: F401, E402


def create_plot(joint_names, duration, checkpoint_name):
    """Create adjacent left/right panels; y-limits are fitted to curve extrema later."""
    preferred = [f"{joint}_{side}_Joint" for joint in ("abad", "hip", "knee", "wheel") for side in ("L", "R")]
    ordered = [name for name in preferred if name in joint_names]
    ordered.extend(name for name in joint_names if name not in ordered)
    indices = [joint_names.index(name) for name in ordered]
    fig, axes = plt.subplots(math.ceil(len(indices) / 2), 2, figsize=(13, 9), sharex=True, squeeze=False)
    lines = []
    for ax, index in zip(axes.flat, indices):
        name = joint_names[index]
        color = "#2471a3" if "_L_" in name else "#c46420"
        line, = ax.plot([], [], color=color, linewidth=0.75)
        lines.append(line)
        ax.axhline(0, color="black", lw=0.4, alpha=0.4)
        ax.set(title=name, ylabel="Torque (N·m)", xlim=(0, duration))
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(indices):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Simulation time (s)")
    fig.suptitle(f"{checkpoint_name} | Joint torque estimates | 200 Hz physics sampling", fontsize=13)
    fig.text(0.5, 0.012, "Implicit PD, post-limit estimate (not measured PhysX drive torque). Gray dotted lines: episode resets.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig, axes, lines, indices


def update_plot(fig, lines, indices, times, torques, resets):
    values = np.array(torques, dtype=float, copy=True)
    # Do not connect different episodes with an artificial torque jump.
    for boundary in resets:
        if boundary < len(values):
            values[boundary] = np.nan
    for line, index in zip(lines, indices):
        line.set_data(times, values[:, index])
        finite = values[:, index][np.isfinite(values[:, index])]
        if finite.size:
            low = float(finite.min())
            high = float(finite.max())
            if low == high:
                low -= 0.05
                high += 0.05
            line.axes.set_ylim(low, high)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def main() -> None:
    checkpoint_path = Path(args_cli.checkpoint_path).expanduser().resolve()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("joint_torques.csv", "joint_torques.png", "summary.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir / name}")

    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    cli_args.configure_getup(env_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    if args_cli.stand_still or args_cli.max_speed:
        # Keep policy inference active and fix the task command across resets
        # and resampling: zero for standing, positive body-y maximum otherwise.
        velocity_cfg = env_cfg.commands.base_velocity
        target_y = float(velocity_cfg.ranges.lin_vel_y[1]) if args_cli.max_speed else 0.0
        velocity_cfg.heading_command = False
        velocity_cfg.ranges.heading = None
        velocity_cfg.rel_heading_envs = 0.0
        velocity_cfg.rel_standing_envs = 1.0 if args_cli.stand_still else 0.0
        velocity_cfg.ranges.lin_vel_x = (0.0, 0.0)
        velocity_cfg.ranges.lin_vel_y = (target_y, target_y)
        velocity_cfg.ranges.ang_vel_z = (0.0, 0.0)
        print(f"[TorqueReplay] Fixed base command: [0, {target_y:g}, 0] (m/s, m/s, rad/s)", flush=True)
    # Camera only: focus on this small robot; no change to commands/dynamics.
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (0.8, 0.8, 0.6)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.12)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)
    print(f"[INFO] Joint order: {joint_names}")
    if args_cli.num_envs != 1:
        print("[WARN] Plotting environment 0 only; use --num_envs 1 for the clearest curve.")

    torque_rows: list[np.ndarray] = []
    time_rows: list[float] = []
    episode_rows: list[int] = []
    command_rows: list[np.ndarray] = []
    resets: list[int] = []
    reset_records = []
    episode = 0
    step_dt = float(env.unwrapped.step_dt)
    physics_dt = float(env.unwrapped.physics_dt)
    max_steps = args_cli.max_steps or math.ceil(args_cli.duration / step_dt)
    duration = max_steps * step_dt
    manager = env.unwrapped.recorder_manager
    original_callback = manager.record_post_physics_decimation_step
    fig = None
    completed_steps = 0
    interrupted = False
    command = None
    started = time.monotonic()

    def record_torques():
        original_callback()
        # These are PD estimates computed BEFORE this physics step, clipped
        # to the effort limits. Implicit PhysX drives receive targets/gains,
        # NOT this estimated torque. Record before automatic episode resets.
        torque_rows.append(robot.data.applied_torque[0].detach().cpu().numpy().copy())
        time_rows.append(len(time_rows) * physics_dt)
        episode_rows.append(episode)
        command_rows.append(command.copy())

    try:
        if not args_cli.zero_actions:
            print(f"[INFO] Loading model checkpoint from: {checkpoint_path}", flush=True)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint_path))
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            encoder = runner.get_inference_encoder(device=env.unwrapped.device)
        obs, obs_dict = env.get_observations()
        obs_history = obs_dict["observations"]["obsHistory"].flatten(start_dim=1)
        commands = obs_dict["observations"]["commands"]
        fig, axes, lines, indices = create_plot(joint_names, duration, checkpoint_path.name)
        fig.suptitle(f"{checkpoint_path.name} | Joint torque estimates | {1 / physics_dt:g} Hz sampling", fontsize=13)
        if not args_cli.headless:
            plt.show(block=False)
            fig.canvas.flush_events()
        manager.record_post_physics_decimation_step = record_torques
        print(f"[TorqueReplay] START: {max_steps} policy steps, {duration:g}s, {1 / physics_dt:g} Hz; output={output_dir}", flush=True)
        started = time.monotonic()
        last_draw = started
        for step in range(max_steps):
            if not simulation_app.is_running():
                interrupted = True
                break
            frame_started = time.monotonic()
            command = env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().numpy().copy()
            with torch.inference_mode():
                if args_cli.zero_actions:
                    actions = torch.zeros((env.num_envs, env.num_actions), device=env.unwrapped.device)
                else:
                    latent = encoder(obs_history)
                    actions = policy(torch.cat((latent, obs, commands), dim=-1).detach())
                obs, _, dones, infos = env.step(actions)
                obs_history = infos["observations"]["obsHistory"].flatten(start_dim=1)
                commands = infos["observations"]["commands"]
            completed_steps = step + 1
            if bool(dones[0]):
                resets.append(len(time_rows))
                reasons = [name for name in env.unwrapped.termination_manager.active_terms
                           if bool(env.unwrapped.termination_manager.get_term(name)[0])]
                reset_records.append({"time_s": completed_steps * step_dt, "episode_id": episode, "reasons": reasons})
                episode += 1
                for ax in axes.flat:
                    ax.axvline(completed_steps * step_dt, color="gray", ls=":", lw=0.5, alpha=0.5)
            if not args_cli.headless and plt.fignum_exists(fig.number) and time.monotonic() - last_draw >= 0.2:
                update_plot(fig, lines, indices, time_rows, torque_rows, resets)
                last_draw = time.monotonic()
            if completed_steps % max(1, round(5 / step_dt)) == 0:
                print(f"[TorqueReplay] {completed_steps * step_dt:.1f}/{duration:g}s; samples={len(time_rows)}, resets={len(resets)}", flush=True)
            if args_cli.real_time or not args_cli.headless:
                time.sleep(max(0, step_dt - (time.monotonic() - frame_started)))
    except KeyboardInterrupt:
        interrupted = True
        print("[TorqueReplay] Interrupted; saving captured samples.", flush=True)
    finally:
        manager.record_post_physics_decimation_step = original_callback
        elapsed_wall = time.monotonic() - started
        try:
            if torque_rows:
                torques = np.asarray(torque_rows, dtype=float)
                if torques.shape != (len(time_rows), len(joint_names)) or not np.isfinite(torques).all():
                    raise RuntimeError("Invalid torque dimensions or non-finite values")
                csv_path = output_dir / "joint_torques.csv"
                with csv_path.open("x", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["time_s", "episode_id", "command_vx_m_s", "command_vy_m_s", "command_wz_rad_s", *joint_names])
                    writer.writerows([t, ep, *cmd, *row] for t, ep, cmd, row in zip(time_rows, episode_rows, command_rows, torques))
                if fig is not None:
                    update_plot(fig, lines, indices, time_rows, torques, resets)
                    plot_path = output_dir / "joint_torques.png"
                    fig.savefig(plot_path, dpi=150)
                    print(f"[TorqueReplay] Plot: {plot_path}", flush=True)
                summary = {
                    "checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    "task": args_cli.task, "seed": env_cfg.seed, "env_id": 0, "num_envs": env.num_envs,
                    "zero_actions": args_cli.zero_actions, "stand_still_command": args_cli.stand_still,
                    "max_speed_command": args_cli.max_speed, "gui": not args_cli.headless,
                    "complete": completed_steps == max_steps and not interrupted,
                    "policy_steps": completed_steps, "physics_samples": len(torque_rows),
                    "duration_s": len(torque_rows) * physics_dt, "policy_dt_s": step_dt, "physics_dt_s": physics_dt,
                    "elapsed_wall_s": elapsed_wall, "reset_records": reset_records,
                    "torque_source": "robot.data.applied_torque: implicit PD post-limit estimate, NOT measured PhysX drive torque",
                    "sample_time": "start of each physics interval; captured before auto-reset",
                    "joint_names": joint_names,
                    "joints": {name: {"min_Nm": float(torques[:, i].min()),
                                     "max_Nm": float(torques[:, i].max()), "peak_abs_Nm": float(np.abs(torques[:, i]).max()),
                                     "rms_Nm": float(np.sqrt(np.mean(torques[:, i] ** 2))),
                                     }
                               for i, name in enumerate(joint_names)},
                }
                with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
                    json.dump(summary, stream, indent=2)
                print(f"[TorqueReplay] CSV: {csv_path}\n[TorqueReplay] Complete={summary['complete']}; samples={len(torque_rows)}", flush=True)
        finally:
            if fig is not None:
                plt.close(fig)
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
