"""Focused jump math and checkpoint migration tests without loading Isaac Sim."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/motor35_jump.py"
EXPAND = ROOT / "scripts/jump/expand_motor35_checkpoint.py"


def load_math():
    tree = ast.parse(MDP.read_text())
    names = {"jump_phase", "reference_pose", "wheel_clearances", "jump_measurements",
             "jump_height_step", "landing_success"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MDP), "exec"), scope)
    return scope


class JumpTests(unittest.TestCase):
    def setUp(self):
        self.fn = load_math()

    def test_reference_is_mirrored_and_returns_to_nominal(self):
        initial = torch.tensor([0., 0., -.3, .3, -.8, .8]).repeat(3, 1)
        pose = self.fn["reference_pose"](torch.tensor([0., .65, 1.3]), initial)
        torch.testing.assert_close(pose[0], initial[0])
        torch.testing.assert_close(pose[2], initial[2])
        torch.testing.assert_close(pose[1, 2::2], -pose[1, 3::2])
        self.assertLess(float(pose[1, 4]), -0.8)

    def test_two_wheels_and_actual_clearance_required(self):
        positions = torch.tensor(
            [[[0., 0., .25], [0., 0., .25]],
             [[0., 0., .25], [0., 0., .25]],
             [[0., 0., .25], [0., 0., .045]]])
        forces = torch.tensor(
            [[[0., 0., 0.], [0., 0., 0.]],
             [[0., 0., 2.], [0., 0., 0.]],
             [[0., 0., 0.], [0., 0., 0.]]])
        scene = {
            "robot": SimpleNamespace(data=SimpleNamespace(body_pos_w=positions)),
            "contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces)),
        }
        class Scene(dict):
            env_origins = torch.zeros(3, 3)
        env = SimpleNamespace(scene=Scene(scene))
        cfg = SimpleNamespace(name="robot", body_ids=[0, 1])
        sensor = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
        height, flying, both = self.fn["jump_measurements"](env, cfg, sensor, .04)
        torch.testing.assert_close(height, torch.tensor([.21, .21, .005]))
        self.assertEqual(flying.tolist(), [True, False, False])
        self.assertEqual(both.tolist(), [False, False, False])

    def test_small_cap_high_uncapped_and_landing_needs_peak(self):
        step = self.fn["jump_height_step"]
        prior = torch.tensor([.10, .10])
        clearance = torch.tensor([.20, .30])
        small, reward = step(prior, clearance, torch.tensor([True, True]), .17, False)
        torch.testing.assert_close(small, clearance)
        torch.testing.assert_close(reward, torch.tensor([.07, .07]) / .17)
        _, reward_high = step(prior, clearance, torch.tensor([True, True]), .17, True)
        self.assertGreater(float(reward_high[1]), float(reward_high[0]))
        landed = self.fn["landing_success"](
            torch.tensor([True, True, True]), torch.tensor([True, True, True]),
            torch.tensor([True, True, False]), torch.tensor([.17, .16, .3]),
            .17, torch.zeros(3), torch.zeros(3))
        self.assertEqual(landed.tolist(), [True, False, False])
        impact = self.fn["landing_success"](
            torch.tensor([True]), torch.tensor([True]), torch.tensor([True]),
            torch.tensor([.18]), .17, torch.tensor([.1]), torch.tensor([-1.5]))
        settled = self.fn["landing_success"](
            torch.tensor([True]), torch.tensor([True]), torch.tensor([True]),
            torch.tensor([.18]), .17, torch.tensor([.1]), torch.tensor([0.]))
        self.assertFalse(bool(impact[0]))
        self.assertTrue(bool(settled[0]))

    def test_checkpoint_preserves_old_columns_and_zeros_new_features(self):
        spec = importlib.util.spec_from_file_location("motor35_expand", EXPAND)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actor = torch.arange(34.0).repeat(2, 1)
        critic = torch.arange(214.0).repeat(2, 1)
        history = torch.arange(280.0).repeat(2, 1)
        ckpt = {"model_state_dict": {"actor.0.weight": actor, "actor.6.weight": torch.zeros(8, 128),
                                     "critic.0.weight": critic},
                "encoder_state_dict": {"encoder.0.weight": history},
                "optimizer_state_dict": {}, "encoder_optimizer_state_dict": {}, "iter": 10000}
        result = module.expand_checkpoint(ckpt)
        a = result["model_state_dict"]["actor.0.weight"]
        c = result["model_state_dict"]["critic.0.weight"]
        h = result["encoder_state_dict"]["encoder.0.weight"].reshape(2, 10, 31)
        torch.testing.assert_close(a[:, :31], actor[:, :31])
        torch.testing.assert_close(a[:, -3:], actor[:, -3:])
        torch.testing.assert_close(a[:, 31:34], torch.zeros(2, 3))
        torch.testing.assert_close(c[:, :211], critic[:, :211])
        torch.testing.assert_close(c[:, -3:], critic[:, -3:])
        torch.testing.assert_close(h[:, :, :28], history.reshape(2, 10, 28))
        torch.testing.assert_close(h[:, :, 28:], torch.zeros(2, 10, 3))
        self.assertEqual(result["iter"], 0)
        self.assertIsNone(result["optimizer_state_dict"])


if __name__ == "__main__":
    unittest.main()
