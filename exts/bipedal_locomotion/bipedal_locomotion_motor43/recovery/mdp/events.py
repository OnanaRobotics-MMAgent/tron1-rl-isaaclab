"""Place the neutral articulation above the plane at a sampled fall orientation."""

import itertools
import math
from pathlib import Path

import torch

from isaaclab.utils.math import quat_apply_inverse, quat_from_angle_axis, quat_mul

from .state import get_state
from .limits import validated_leg_limits


def prepare_getup(env, env_ids):
    """Read neutral collision bounds once, including instance proxies / guide geometry.

    The USD's authored articulation must match the zero-joint reset. Bounding-box
    corners conservatively enclose each collider; rotating only the root while
    keeping neutral joints therefore cannot initialize a collider below the plane.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    asset = env.scene["robot"]
    leg_ids, names = asset.find_joints(list(env.cfg.getup.leg_joint_limits))
    limits = validated_leg_limits(names, env.cfg.getup.leg_joint_limits,
                                  asset.data.joint_pos_limits[:, leg_ids])
    # An actual PhysX constraint, independent of reward weights or policy outputs.
    # Isaac Sim 5.x rejects the +/- infinity limits used by continuous wheel
    # joints when the articulation-wide limit buffer is submitted. Recovery
    # does not need multiple wheel revolutions, so use a finite temporary
    # range for the wheel DOFs while preserving the real leg limits.
    physx_limits = asset.data.joint_pos_limits.clone()
    physx_limits[:, leg_ids] = limits
    leg_id_set = {int(idx) for idx in leg_ids}
    wheel_ids = [idx for idx in range(asset.num_joints) if idx not in leg_id_set]
    if wheel_ids:
        physx_limits[:, wheel_ids, 0] = -2.0 * math.pi
        physx_limits[:, wheel_ids, 1] = 2.0 * math.pi
    asset.write_joint_position_limit_to_sim(physx_limits)
    env._getup_leg_joint_ids = leg_ids
    if torch.any(asset.data.default_joint_pos.abs() > 1.0e-6):
        raise ValueError("GetUp collision bounds require the WF_TRON1A zero-joint default pose.")
    stage = Usd.Stage.Open(asset.cfg.spawn.usd_path)
    root = stage.GetPrimAtPath(str(stage.GetDefaultPrim().GetPath()) + "/base_Link")
    root_inv = UsdGeom.XformCache().GetLocalToWorldTransform(root).GetInverse()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    corners = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bounds.IsEmpty():
                continue
            lo, hi = bounds.GetMin(), bounds.GetMax()
            for indices in itertools.product((0, 1), repeat=3):
                point = Gf.Vec3d(*[(lo, hi)[indices[i]][i] for i in range(3)])
                corners.append(tuple(root_inv.Transform(point)))
    if not corners:
        # IsaacLab's current URDF converter keeps collision geometry in a
        # sibling physics layer instead of composing it into the main asset.
        usd_path = Path(asset.cfg.spawn.usd_path)
        physics_path = usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd"
        physics_stage = Usd.Stage.Open(str(physics_path))
        if physics_stage:
            physics_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
            for prim in physics_stage.Traverse():
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    bounds = physics_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    if bounds.IsEmpty():
                        continue
                    lo, hi = bounds.GetMin(), bounds.GetMax()
                    for indices in itertools.product((0, 1), repeat=3):
                        point = Gf.Vec3d(*[(lo, hi)[indices[i]][i] for i in range(3)])
                        corners.append(tuple(point))
    if not corners:
        raise RuntimeError("WF_TRON1A collision geometry could not be read.")
    env._getup_collision_corners = torch.tensor(corners, device=env.device, dtype=torch.float32)
    get_state(env)


def reset_fallen(env, env_ids):
    state = get_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    cfg = env.cfg.getup
    level = torch.full((n,), state.level, device=env.device, dtype=torch.long)
    if cfg.curriculum_enabled and state.level > 0:
        replay = torch.rand(n, device=env.device) < cfg.replay_probability
        level[replay] = torch.randint(state.level, (int(replay.sum()),), device=env.device)
    ranges = torch.tensor(cfg.tilt_ranges_deg, device=env.device)
    tilt = (ranges[level, 0] + torch.rand(n, device=env.device) *
            (ranges[level, 1] - ranges[level, 0])) * math.pi / 180.0
    # Stage 2: front/back. Stage 3: front/back/left/right, plus 20% overturned.
    direction = torch.randint(4, (n,), device=env.device) * (math.pi / 2)
    front_back = torch.randint(2, (n,), device=env.device) * math.pi + math.pi / 2
    direction = torch.where(level == getattr(cfg, "front_back_level", 2), front_back, direction)
    state.direction[env_ids] = torch.round(direction / (math.pi / 2)).long() % 4
    overturned = (level == getattr(cfg, "overturned_level", 3)) & (torch.rand(n, device=env.device) < 0.2)
    tilt[overturned] = math.pi * (0.75 + 0.25 * torch.rand(int(overturned.sum()), device=env.device))
    axis = torch.stack((torch.cos(direction), torch.sin(direction), torch.zeros_like(direction)), dim=-1)
    rotation = quat_from_angle_axis(tilt, axis)
    yaw = (2 * torch.rand(n, device=env.device) - 1) * math.pi
    yaw_axis = torch.zeros(n, 3, device=env.device)
    yaw_axis[:, 2] = 1.0
    rotation = quat_mul(quat_from_angle_axis(yaw, yaw_axis), rotation)

    corners = env._getup_collision_corners
    # Only vertical support is needed; avoid allocating n x vertices x 3
    # rotated geometry (especially for the new nominal-pose convex hulls).
    up = torch.zeros(n, 3, device=env.device)
    up[:, 2] = 1.0
    local_up = quat_apply_inverse(rotation, up)
    bottom = (local_up @ corners.T).amin(dim=1)
    asset = env.scene["robot"]
    root = asset.data.default_root_state[env_ids].clone()
    root[:, :3] = env.scene.env_origins[env_ids]
    root[:, :2] += (torch.rand(n, 2, device=env.device) - 0.5) * 0.3
    root[:, 2] += (-bottom).clamp(
        min=getattr(cfg, "min_root_height", 0.12)) + cfg.reset_clearance
    root[:, 3:7] = rotation
    root[:, 7:13] = 0.0
    # Ground starts, not high-altitude drops; never step all environments from a reset callback.
    asset.write_joint_state_to_sim(asset.data.default_joint_pos[env_ids],
                                  torch.zeros_like(asset.data.default_joint_vel[env_ids]), env_ids=env_ids)
    asset.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    state.stage[env_ids] = level
    state.hold[env_ids] = 0
    state.steps[env_ids] = 0
    state.success[env_ids] = False
    state.active[env_ids] = True
    state.landing_steps[env_ids] = 0
    state.control_ready[env_ids] = not getattr(cfg, "require_landing", False)
    state.start_xy[env_ids] = root[:, :2]
