"""Generate train/test pose banks using passive articulated falls, without PPO."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'exts/bipedal_locomotion'))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', required=True)
p.add_argument('--train_poses', type=int, default=4096)
p.add_argument('--test_poses', type=int, default=1024)
p.add_argument('--num_envs', type=int, default=128)
p.add_argument('--seed', type=int, default=20260918)
p.add_argument('--settle_seconds', type=float, default=5.0)
p.add_argument('--max_batches', type=int, default=200)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if min(a.train_poses, a.test_poses, a.num_envs, a.max_batches) <= 0 or a.settle_seconds < 1:
    p.error('Positive counts and at least one settling second are required')
out = Path(a.output).resolve()
if out.exists():
    p.error(f'Output already exists: {out}')
app = AppLauncher(a).app
import hashlib
import json
import math
import torch
import gymnasium as gym
from pxr import Usd, UsdGeom
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul
import bipedal_locomotion_motor43, bipedal_locomotion_motor35
from bipedal_locomotion_motor43.recovery.mdp.pose_bank import pose_ids, validate_bank

cfg = parse_env_cfg('Isaac-Motor43-FallenPose-Generate-v0', device=a.device, num_envs=a.num_envs)
cfg.seed = a.seed
env = gym.make('Isaac-Motor43-FallenPose-Generate-v0', cfg=cfg).unwrapped
robot = env.scene['robot']
leg_ids = env._getup_leg_joint_ids
limits = robot.data.joint_pos_limits[0, leg_ids]

# Collision support points expressed in their owning rigid link frames.
# Cylinders use 128 rim samples (sub-millimeter floor approximation).
# Recompute world positions from simulated link transforms for each new joint pose.
stage = Usd.Stage.Open(robot.cfg.spawn.usd_path)
from pxr import UsdPhysics, Gf
xforms = UsdGeom.XformCache()
vertices, body_indices = [], []
for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
    if not any(prim.IsA(kind) for kind in (UsdGeom.Mesh, UsdGeom.Cube, UsdGeom.Cylinder)):
        continue
    owner = prim
    while owner and not owner.HasAPI(UsdPhysics.CollisionAPI):
        owner = owner.GetParent()
    if not owner:
        continue
    parent = prim
    while parent and parent.GetName() not in robot.body_names:
        parent = parent.GetParent()
    if not parent:
        continue
    transform = xforms.GetLocalToWorldTransform(prim) * xforms.GetLocalToWorldTransform(parent).GetInverse()
    if prim.IsA(UsdGeom.Mesh):
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    elif prim.IsA(UsdGeom.Cube):
        import itertools
        half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
        points = list(itertools.product((-half, half), repeat=3))
    else:
        shape = UsdGeom.Cylinder(prim)
        radius, height = shape.GetRadiusAttr().Get(), shape.GetHeightAttr().Get()
        axis = {'X': 0, 'Y': 1, 'Z': 2}[shape.GetAxisAttr().Get()]
        other = [i for i in range(3) if i != axis]
        points = []
        for cap in (-height/2, height/2):
            for k in range(128):
                xyz = [0., 0., 0.];xyz[axis] = cap
                xyz[other[0]] = radius * math.cos(2*math.pi*k/128)
                xyz[other[1]] = radius * math.sin(2*math.pi*k/128)
                points.append(xyz)
    for point in points:
        vertices.append(tuple(transform.Transform(Gf.Vec3d(*point))))
        body_indices.append(robot.body_names.index(parent.GetName()))
if not vertices:
    raise RuntimeError('No collision mesh vertices found')
vertices = torch.tensor(vertices, device=env.device, dtype=torch.float)
body_indices = torch.tensor(body_indices, device=env.device)
print('Collision vertices:', len(vertices), 'joint order:', robot.joint_names, flush=True)


def floor_clearance():
    pos = robot.data.body_link_pos_w[:, body_indices]
    quat = robot.data.body_link_quat_w[:, body_indices]
    pts = quat_apply(quat.reshape(-1, 4), vertices.repeat(a.num_envs, 1)).reshape(a.num_envs, -1, 3) + pos
    return pts[:, :, 2].amin(1) - env.scene.env_origins[:, 2]


seen = set()
rejected = dict(unstable=0, upright=0, below_ground=0, limits=0, duplicate=0)

def generate(count, seed):
    env.seed(seed)
    records = []
    batch_ids = []
    accepted = 0
    for batch in range(a.max_batches):
        root = robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins
        root[:, 2] += 2.0  # conservative clearance; record only settled ground states
        root[:, 7:] = 0
        direction = torch.randint(4, (a.num_envs,), device=env.device)
        theta = direction * math.pi / 2
        axis = torch.stack((theta.cos(), theta.sin(), torch.zeros_like(theta)), 1)
        rotation = quat_from_angle_axis(torch.rand(a.num_envs, device=env.device) * math.radians(120) + math.radians(60), axis)
        yaw_axis = torch.zeros(a.num_envs, 3, device=env.device); yaw_axis[:, 2] = 1
        root[:, 3:7] = quat_mul(quat_from_angle_axis(torch.rand(a.num_envs, device=env.device) * 2 * math.pi, yaw_axis), rotation)
        q = robot.data.default_joint_pos.clone()
        q[:, leg_ids] = limits[:, 0] + torch.rand(a.num_envs, len(leg_ids), device=env.device) * (limits[:, 1] - limits[:, 0])
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.scene.reset()
        robot.set_joint_position_target(q)
        robot.set_joint_velocity_target(torch.zeros_like(q))
        stable_count = torch.zeros(a.num_envs, device=env.device, dtype=torch.long)
        # No env.step: no policy, reward, auto-reset or curriculum during generation.
        for step in range(round(a.settle_seconds / env.step_dt)):
            for _ in range(env.cfg.decimation):
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
            data = robot.data
            stable = ((data.root_lin_vel_w.norm(dim=1) < .10)
                      & (data.root_ang_vel_w.norm(dim=1) < .20)
                      & (data.joint_vel[:, leg_ids].abs().amax(1) < .35))
            stable_count = torch.where(stable, stable_count + 1, 0)
        data = robot.data
        root = data.root_state_w.clone();root[:, :3] -= env.scene.env_origins
        # Translational invariance: center snapshots without changing ground height.
        root[:, :2] = 0
        q = data.joint_pos.clone(); qd = data.joint_vel.clone()
        gravity = data.projected_gravity_b
        tilt = torch.acos((-gravity[:, 2]).clamp(-1, 1))
        direction = torch.round(torch.atan2(gravity[:, 0], -gravity[:, 1]) / (math.pi / 2)).long() % 4
        floor = floor_clearance()
        stable = stable_count >= round(.5 / env.step_dt)
        fallen = tilt > math.radians(55)
        floor_ok = floor >= -.02
        joints_ok = ((q[:, leg_ids] >= limits[:, 0] - .025) & (q[:, leg_ids] <= limits[:, 1] + .025)).all(1)
        force = env.scene['contact_forces'].data.net_forces_w.norm(dim=-1)
        valid = (stable & fallen & floor_ok & joints_ok & (force.amax(1) > 5)
                 & (force.amax(1) < 2000) & (root[:, 2] < 1.2)
                 & torch.isfinite(root).all(1) & torch.isfinite(q).all(1))
        rejected['unstable'] += int((~stable).sum());rejected['upright'] += int((~fallen).sum())
        rejected['below_ground'] += int((~floor_ok).sum());rejected['limits'] += int((~joints_ok).sum())
        difficulty = ((q[:, leg_ids] / torch.maximum(limits[:, 0].abs(), limits[:, 1].abs())).square().mean(1)).sqrt()
        indices = valid.nonzero().flatten()
        signatures = pose_ids(root[indices], q[indices])
        for idx, signature in zip(indices.tolist(), signatures):
            if signature in seen:
                rejected['duplicate'] += 1;continue
            seen.add(signature)
            records.append(dict(root_state=root[idx].cpu(), joint_pos=q[idx].cpu(), joint_vel=qd[idx].cpu(),
                                direction=direction[idx].cpu(), difficulty=difficulty[idx].cpu(),
                                tilt_deg=torch.rad2deg(tilt[idx]).cpu(), min_floor_z=floor[idx].cpu()))
            accepted += 1
            if accepted == count:
                break
        print(f'seed={seed} batch={batch} accepted={accepted}/{count} stable={int(stable.sum())} floor_ok={int(floor_ok.sum())}', flush=True)
        if accepted == count:
            return {key: torch.stack([row[key] for row in records]) for key in records[0]}
    raise RuntimeError(f'Only {accepted}/{count} valid poses; increase max_batches or inspect filters')

try:
    train = generate(a.train_poses, a.seed)
    test = generate(a.test_poses, a.seed + 1)
    thresholds = torch.quantile(train['difficulty'], torch.tensor([1/3, 2/3]))
    for data in (train, test):
        data['stage'] = torch.bucketize(data['difficulty'], thresholds).long()
    bank = dict(version=1, joint_names=robot.joint_names, train=train, test=test,
                thresholds=thresholds, seed=a.seed, leg_joint_ids=leg_ids, leg_limits=limits.cpu(),
                asset_sha256=hashlib.sha256(Path(robot.cfg.spawn.usd_path).read_bytes()).hexdigest(),
                generation=dict(passive=True, settle_seconds=a.settle_seconds, stable_hold_seconds=.5,
                                max_linear_speed=.1, max_angular_speed=.2, max_leg_speed=.35,
                                floor_tolerance=.02, rejected=rejected))
    validate_bank(bank)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp');torch.save(bank, tmp);tmp.replace(out)
    summary = {k:v for k,v in bank.items() if k not in ('train','test','thresholds','leg_limits')}
    summary['thresholds'] = thresholds.tolist()
    summary['counts'] = {split: [[int(((data['stage']==s)&(data['direction']==d)).sum()) for d in range(4)] for s in range(3)]
                         for split,data in [('train',train),('test',test)]}
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
finally:
    env.close();app.close()
