import copy
import importlib.util
from pathlib import Path
import unittest
import torch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pose_bank',ROOT/'exts/bipedal_locomotion/bipedal_locomotion_motor43/recovery/mdp/pose_bank.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def bank():
    def split(offset):
        root=torch.zeros(3,13);root[:,3]=1;root[:,2]=.4
        q=torch.tensor([[.1],[.4],[.8]])+offset
        d=torch.tensor([.1,.4,.8])
        return dict(root_state=root,joint_pos=q,joint_vel=torch.zeros_like(q),difficulty=d,
                    direction=torch.tensor([0,1,2]),stage=torch.tensor([0,1,2]))
    return dict(version=1,joint_names=['leg'],leg_joint_ids=[0],leg_limits=torch.tensor([[-1.,1.]]),
                thresholds=torch.tensor([.3,.6]),train=split(0),test=split(.02))


class TestPoseBank(unittest.TestCase):
    def test_train_test_are_independent(self):
        data=bank();m.validate_bank(data)
        data['test']=copy.deepcopy(data['train'])
        with self.assertRaisesRegex(ValueError,'leakage'):m.validate_bank(data)
    def test_limits_and_quaternions(self):
        data=bank();data['test']['joint_pos'][0]=7
        with self.assertRaisesRegex(ValueError,'limits'):m.validate_bank(data)
        data=bank();data['test']['root_state'][0,3]=0
        with self.assertRaisesRegex(ValueError,'quaternion'):m.validate_bank(data)
    def test_stage_mapping_and_joint_order(self):
        data=bank();data['test']['stage'][0]=2
        with self.assertRaisesRegex(ValueError,'bands'):m.validate_bank(data)
        with self.assertRaisesRegex(ValueError,'joint order'):m.validate_bank(bank(),['wheel'])
