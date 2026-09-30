"""Short physics preflight, never a policy quality/success-rate evaluation."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/bipedal_locomotion"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Motor35-Locomotion-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--collision_probe", action="store_true", help="Nominal pose in free space without gravity")
parser.add_argument("--output", type=Path, default=ROOT / "outputs/motor35/physics_check.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("--num_envs and --steps must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import torch
import bipedal_locomotion_motor43, bipedal_locomotion_motor35  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply
from bipedal_locomotion_motor35.assets.config import motor35_parameters as p
from bipedal_locomotion_motor35.assets.config.motor35_cfg import PREPARED
from bipedal_locomotion_motor35.mdp import read_physical_properties


def main():
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"task": args.task, "status": "running"}) + "\n")
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
    if args.collision_probe:
        if "Recovery" in args.task:
            raise ValueError("Collision probe uses Locomotion-Play, before enabling fallen resets")
        cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        cfg.scene.robot.init_state.pos = (0., 0., 1.)
        cfg.terminations.base_contact = None
        cfg.terminations.non_wheel_contact = None
    env = gym.make(args.task, cfg=cfg)
    result = {"task": args.task, "collision_probe": args.collision_probe, "num_envs": args.num_envs}
    try:
        raw = env.unwrapped
        obs, _ = env.reset()
        robot = raw.scene["robot"]
        # Importer must preserve CAD inertials rather than recalculate them
        # from the collision decomposition. Compare principal moments because
        # PhysX stores the tensor in its mass frame.
        model = ET.parse(PREPARED).getroot()
        authored = {l.get("name"): l for l in model.findall("link")}
        for i, name in enumerate(robot.body_names):
            link = authored[name]
            mass = float(link.find("inertial/mass").get("value"))
            torch.testing.assert_close(robot.data.default_mass[0, i].cpu(), torch.tensor(mass), atol=1e-6, rtol=1e-5)
            a = {k: float(v) for k, v in link.find("inertial/inertia").attrib.items()}
            matrix = torch.tensor([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]],
                                   [a["ixz"], a["iyz"], a["izz"]]])
            actual = robot.data.default_inertia[0, i].cpu().reshape(3, 3)
            torch.testing.assert_close(torch.linalg.eigvalsh(actual), torch.linalg.eigvalsh(matrix), atol=1e-8, rtol=1e-3)
        result["imported_cad_inertials_match"] = True
        result["joint_names"] = robot.joint_names
        result["observation_shapes"] = {k: list(v.shape[1:]) for k, v in obs.items()}
        assert list(obs["policy"].shape[1:]) == [28]
        assert list(obs["obsHistory"].shape[1:]) == [10, 28]
        assert list(obs["commands"].shape[1:]) == [3]
        actual_properties = read_physical_properties(raw)
        torch.testing.assert_close(actual_properties, raw._motor35_physical_properties)
        assert actual_properties.shape[1] == 142
        masses = robot.root_physx_view.get_masses()
        ratios = masses / robot.data.default_mass.cpu()
        assert (ratios >= .89999).all() and (ratios <= 1.10001).all()
        result["mass_range_kg"] = [float(masses.sum(1).min()), float(masses.sum(1).max())]
        assert torch.count_nonzero(robot.actuators["wheels"].stiffness) == 0
        wheel_ids, _ = robot.find_joints("wheel_[LR]_Joint")
        wheel_limits = robot.data.joint_pos_limits[:, wheel_ids]
        print("[Motor35 preflight] wheel limits:", wheel_limits[0].cpu().tolist(), flush=True)
        # PhysX may represent continuous joints with +/- float max, not inf.
        assert (wheel_limits[..., 0] < -1e6).all() and (wheel_limits[..., 1] > 1e6).all()
        result["wheel_limits_unbounded"] = True
        result["motor_peak_nm"] = p.PEAK_TORQUE_NM
        result["self_collision"] = True
        stages = [None] if not hasattr(cfg, "getup") else [0, 18, 35]
        max_torque = max_contact = max_pose_drift = 0.
        done_count = 0
        for stage in stages:
            if stage is not None:
                raw.task_state.level = stage
                cfg.getup.curriculum_enabled = False
                obs, _ = env.reset()
                corners = raw._getup_collision_corners
                rotation = robot.data.root_quat_w[:, None, :].expand(-1, len(corners), -1)
                points = quat_apply(rotation.reshape(-1, 4), corners.expand(args.num_envs, -1, -1).reshape(-1, 3))
                minimum = points.reshape(args.num_envs, -1, 3)[:, :, 2].amin(1)
                minimum += robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                assert (minimum >= cfg.getup.reset_clearance - 1e-5).all()
            for _ in range(args.steps):
                actions = torch.zeros(args.num_envs, 8, device=raw.device)
                if not args.collision_probe:
                    actions.normal_(mean=0., std=.03)
                obs, reward, terminated, truncated, _ = env.step(actions)
                assert all(torch.isfinite(v).all() for v in obs.values())
                assert torch.isfinite(reward).all()
                torque = float(robot.data.applied_torque.abs().max())
                assert torque <= p.PEAK_TORQUE_NM + 1e-5
                max_torque = max(max_torque, torque)
                max_contact = max(max_contact, float(raw.scene["contact_forces"].data.net_forces_w.norm(dim=-1).max()))
                max_pose_drift = max(max_pose_drift, float((robot.data.joint_pos - robot.data.default_joint_pos).abs().max()))
                done_count += int((terminated | truncated).sum())
        result.update(max_applied_torque_nm=max_torque, max_net_contact_n=max_contact,
                      max_joint_pose_drift_rad=max_pose_drift, reset_count=done_count, stages=stages,
                      steps_per_stage=args.steps)
        if args.collision_probe:
            # No gravity/ground/actions: any substantial forces or leg motion
            # indicate initial collision overlap, not a weak balance policy.
            result["collision_probe_passed"] = max_contact < 1. and max_pose_drift < .02
        else:
            result["finite_rollout_passed"] = True
        result["status"] = "failed" if args.collision_probe and not result["collision_probe_passed"] else "passed"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("[Motor35 preflight] " + json.dumps(result))
        if args.collision_probe and not result["collision_probe_passed"]:
            raise RuntimeError("Initial self-collision overlap remains; inspect physics_check.json before training")
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException as error:
        import traceback
        traceback.print_exc()
        result = json.loads(args.output.read_text()) if args.output.exists() else {}
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        sys.stdout.flush()
        sys.stderr.flush()
        exit_code = 1
    finally:
        # Keep Kit's normal fast shutdown (full plugin unloading can crash),
        # but preserve failures instead of letting shutdown report success.
        app.app.post_quit(exit_code)
        app.close()
    sys.exit(exit_code)
