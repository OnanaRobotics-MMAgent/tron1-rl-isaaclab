"""Validated settled articulation states, independent of the simulator."""
from pathlib import Path
import hashlib
import torch


def pose_ids(root, joints):
    # Quantization removes exact/near duplicates within and across dataset splits.
    values = torch.cat((root[:, 2:7], joints), -1).cpu()
    return [hashlib.sha256(row.numpy().tobytes()).hexdigest()
            for row in (values * 1000).round().to(torch.int32)]


def validate_bank(bank, joint_names=None):
    if bank.get('version') != 1:
        raise ValueError('Unsupported fallen-pose bank version')
    if joint_names is not None and list(joint_names) != bank['joint_names']:
        raise ValueError('Pose bank joint order does not match the robot')
    seen = set()
    for name in ('train', 'test'):
        data = bank[name]
        n = len(data['root_state'])
        if not n or data['root_state'].shape != (n, 13):
            raise ValueError(f'Invalid {name} root states')
        for key in ('root_state', 'joint_pos', 'joint_vel', 'direction', 'difficulty', 'stage'):
            if len(data[key]) != n or not bool(torch.isfinite(data[key]).all()):
                raise ValueError(f'Invalid {name}/{key}')
        if data['joint_pos'].shape != (n, len(bank['joint_names'])) or data['joint_vel'].shape != data['joint_pos'].shape:
            raise ValueError('Invalid articulation dimensions')
        if not bool(torch.allclose(data['root_state'][:, 3:7].norm(dim=1), torch.ones(n), atol=1e-3)):
            raise ValueError('Non-unit root quaternion')
        ids = pose_ids(data['root_state'], data['joint_pos'])
        if len(set(ids)) != n or seen.intersection(ids):
            raise ValueError('Duplicate poses or train/test leakage')
        seen.update(ids)
        if not bool(((data['direction'] >= 0) & (data['direction'] < 4)).all()):
            raise ValueError('Invalid direction')
        if not bool(((data['stage'] >= 0) & (data['stage'] < 3)).all()):
            raise ValueError('Invalid pose stage')
        limits = bank['leg_limits']
        q = data['joint_pos'][:, bank['leg_joint_ids']]
        if not bool(((q >= limits[:, 0] - .025) & (q <= limits[:, 1] + .025)).all()):
            raise ValueError('Pose bank exceeds physical leg limits')
        if not torch.equal(data['stage'], torch.bucketize(data['difficulty'], bank['thresholds'])):
            raise ValueError('Pose stage does not match the deformation bands')
    return bank


def load_bank(path, joint_names=None):
    bank = torch.load(Path(path), map_location='cpu', weights_only=False)
    return validate_bank(bank, joint_names)
