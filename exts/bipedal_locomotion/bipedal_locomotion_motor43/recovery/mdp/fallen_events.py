"""Restore settled random-joint poses without stepping other environments."""
import torch
import hashlib
from pathlib import Path
from .events import prepare_getup, reset_fallen
from .state import get_state
from .pose_bank import load_bank


def prepare_fallen(env, env_ids):
    prepare_getup(env, env_ids)
    cfg = env.cfg.fallen
    bank = load_bank(cfg.bank_path, env.scene['robot'].joint_names)
    if hashlib.sha256(Path(env.scene['robot'].cfg.spawn.usd_path).read_bytes()).hexdigest() != bank['asset_sha256']:
        raise ValueError('Pose bank was generated with a different robot asset')
    data = bank[cfg.split]
    env._fallen_bank = {k: v.to(env.device) for k, v in data.items() if isinstance(v, torch.Tensor)}
    env._fallen_indices = []
    # Direction-balanced sampling, with stage determined only by joint deformation.
    for stage in range(3):
        env._fallen_indices.append([])
        for direction in range(4):
            ids = ((data['stage'] == stage) & (data['direction'] == direction)).nonzero().flatten()
            if not len(ids):
                raise ValueError(f'Pose bank is empty at stage {stage}, direction {direction}')
            env._fallen_indices[-1].append(ids.to(env.device))
    env.fallen_pose_id = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)


def reset_from_bank(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if not n:
        return
    state = get_state(env)
    cfg = env.cfg.fallen
    force = getattr(env, '_fallen_forced_indices', None)
    if force is not None:
        ids = force[env_ids]
        replay = torch.zeros(n, device=env.device, dtype=torch.bool)
    else:
        replay = torch.rand(n, device=env.device) < cfg.old_pose_probability
        levels = torch.full((n,), state.level, device=env.device, dtype=torch.long)
        if env.cfg.getup.curriculum_enabled and state.level > 0:
            easier = torch.rand(n, device=env.device) < 0.15
            levels[easier] = torch.randint(state.level, (int(easier.sum()),), device=env.device)
        direction = torch.randint(4, (n,), device=env.device)
        ids = torch.empty(n, device=env.device, dtype=torch.long)
        for level in range(3):
            for d in range(4):
                mask = (levels == level) & (direction == d)
                pool = env._fallen_indices[level][d]
                ids[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), device=env.device)]
    if replay.any():
        old_ids = env_ids[replay]
        reset_fallen(env, old_ids)
        state.stage[old_ids] = -1  # exclude old-pose rehearsal from promotion
        env.fallen_pose_id[old_ids] = -1
    target = env_ids[~replay]
    ids = ids[~replay]
    if not len(target):
        return
    data = env._fallen_bank
    root = data['root_state'][ids].clone()
    root[:, :3] += env.scene.env_origins[target]
    robot = env.scene['robot']
    robot.write_joint_state_to_sim(data['joint_pos'][ids], data['joint_vel'][ids], env_ids=target)
    robot.write_root_state_to_sim(root, env_ids=target)
    # No stale position drive target while reset observations are constructed.
    robot.set_joint_position_target(data['joint_pos'][ids], env_ids=target)
    robot.set_joint_velocity_target(torch.zeros_like(data['joint_vel'][ids]), env_ids=target)
    state.stage[target] = data['stage'][ids]
    state.direction[target] = data['direction'][ids]
    state.hold[target] = 0
    state.steps[target] = 0
    state.success[target] = False
    state.active[target] = True
    state.start_xy[target] = root[:, :2]
    env.fallen_pose_id[target] = ids
