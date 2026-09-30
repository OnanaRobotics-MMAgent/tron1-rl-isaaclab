"""Source-model regression checks; run with Isaac Sim python.sh, without Kit."""
import ast
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prepare_motor35", ROOT / "tools/prepare_motor35.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TestMotor35(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        cls.report = prepare.prepare(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_si_inertia_matches_cad_and_whole_robot_mass(self):
        self.assertTrue(self.report["cad_inertia_and_com_match"])
        self.assertAlmostEqual(self.report["total_mass_kg"], 2.621)

    def test_preserves_joint_axes_limits_inertia_and_meshes(self):
        src = ET.parse(prepare.SOURCE).getroot()
        dst = ET.parse(self.path / "robot.urdf").getroot()
        targets = {x.get("name"): x for x in dst.findall("joint")}
        for joint in src.findall("joint"):
            other = targets[prepare.motor.JOINT_MAP[joint.get("name")]]
            for tag in ("axis", "limit", "origin"):
                self.assertEqual(joint.find(tag).attrib, other.find(tag).attrib)
            self.assertEqual(joint.get("type"), other.get("type"))
        targets = {x.get("name"): x for x in dst.findall("link")}
        for link in src.findall("link"):
            other = targets[prepare.motor.LINK_MAP[link.get("name")]]
            for tag in ("origin", "mass", "inertia"):
                self.assertEqual(link.find(f"inertial/{tag}").attrib, other.find(f"inertial/{tag}").attrib)
        for mesh in dst.findall(".//mesh"):
            self.assertTrue((self.path / mesh.get("filename")).exists())

    def test_fk_reference_and_new_forward_axis(self):
        self.assertEqual(self.report["forward_axis"], "x")
        self.assertAlmostEqual(self.report["nominal_com_base_m"][0], self.report["wheel_center_base_m"][0])
        self.assertLess(prepare.motor.MAX_COMMAND_SPEED_MPS,
                        prepare.motor.WHEEL_RADIUS_M * prepare.motor.RATED_SPEED_RAD_S)
        # Old 2 m/s request exceeds even this model's no-load wheel speed.
        self.assertGreater(2., prepare.motor.WHEEL_RADIUS_M * prepare.motor.NO_LOAD_SPEED_RAD_S)

    def test_reports_false_contacts_from_single_hulls(self):
        self.assertEqual(len(self.report["naive_convex_hull_overlaps"]), 2)


class TestSelectedMotorPower(unittest.TestCase):
    def test_subset_and_absolute_power_do_not_cancel_or_include_wheels(self):
        path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion_motor43/locomotion/mdp/rewards.py"
        f = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "joint_powers_l1")
        scope = {"torch": torch, "ManagerBasedRLEnv": object,
                 "SceneEntityCfg": lambda name: SimpleNamespace(name=name, joint_ids=slice(None))}
        exec(compile(ast.Module(body=[f], type_ignores=[]), str(path), "exec"), scope)
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
            applied_torque=torch.tensor([[1., -2., 100.], [-1., 2., 100.]]),
            joint_vel=torch.tensor([[3., 4., 100.], [-3., -4., 100.]])))})
        fn = scope["joint_powers_l1"]
        torch.testing.assert_close(fn(env, SimpleNamespace(name="robot", joint_ids=[0, 1])), torch.tensor([11., 11.]))
        torch.testing.assert_close(fn(env), torch.tensor([10011., 10011.]))


class TestCheckpointContract(unittest.TestCase):
    def test_rejects_wrong_model_before_loading_any_network(self):
        path = ROOT / "rsl_rl/rsl_rl/runner/on_policy_runner.py"
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "OnPolicyRunner")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load")
        checkpoint = {"model_state_dict": {}, "encoder_state_dict": {}, "iter": 0, "infos": None}
        scope = {"torch": SimpleNamespace(load=lambda *args, **kwargs: checkpoint)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
        raw = SimpleNamespace(cfg=SimpleNamespace(policy_contract={"robot": "Motor35", "revision": "a"}))
        runner = SimpleNamespace(device="cpu", env=SimpleNamespace(unwrapped=raw), alg=Mock())
        for contract in (None, {"robot": "Motor35", "revision": "b"}):
            checkpoint["policy_contract"] = contract
            with self.assertRaisesRegex(ValueError, "contract differs"):
                scope["load"](runner, "unused.pt")
            runner.alg.actor_critic.load_state_dict.assert_not_called()
        checkpoint["policy_contract"] = raw.cfg.policy_contract.copy()
        scope["load"](runner, "unused.pt")
        runner.alg.actor_critic.load_state_dict.assert_called_once()
        # Untagged legacy checkpoints retain their existing behavior.
        raw.cfg = SimpleNamespace()
        checkpoint.pop("policy_contract")
        scope["load"](runner, "unused.pt")


if __name__ == "__main__":
    unittest.main()
