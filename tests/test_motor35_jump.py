"""Motor35 FK/IK, reference jump state/rewards and checkpoint regressions."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/motor35_jump.py"
EXPAND = ROOT / "scripts/jump/expand_motor35_checkpoint.py"
URDF = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/Motor35_WF.urdf"


def load_math():
    tree = ast.parse(MDP.read_text())
    nodes = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    scope = {"torch": torch, "math": math, "JointPositionAction": object,
             "ManagerTermBase": object, "configclass": lambda cls: cls}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MDP), "exec"), scope)
    return scope


class Scene(dict):
    def __init__(self, values, n):
        super().__init__(values)
        self.env_origins = torch.zeros(n, 3)


def make_env(fn, n=2):
    q = torch.tensor([0., 0., -.3, .3, -.8, .8]).repeat(n, 1)
    data = SimpleNamespace(joint_pos=q, default_joint_pos=q.clone(),
        body_pos_w=torch.tensor([[[0., 0., .04], [0., 0., .04]]]).repeat(n, 1, 1),
        root_pos_w=torch.tensor([[0., 0., .2]]).repeat(n, 1),
        root_lin_vel_w=torch.zeros(n, 3), root_lin_vel_b=torch.zeros(n, 3),
        root_ang_vel_b=torch.zeros(n, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1))
    robot = SimpleNamespace(data=data, find_joints=lambda *a, **kw: (list(range(6)), []),
                            find_bodies=lambda *a, **kw: ([0, 1], []))
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.ones(n, 2, 3)*5),
                             find_bodies=lambda *a, **kw: ([0, 1], []))
    action = SimpleNamespace(raw_actions=torch.zeros(n, 6), processed_actions=q.clone())
    cfg = fn["JumpCfg"]()
    return SimpleNamespace(num_envs=n, device="cpu", cfg=SimpleNamespace(jump=cfg), step_dt=.02,
        scene=Scene({"robot": robot, "contact_forces": sensor}, n),
        action_manager=SimpleNamespace(get_term=lambda name: action))


class JumpTests(unittest.TestCase):
    def setUp(self):
        self.fn = load_math()

    def test_urdf_geometry_and_reference_joint_limits(self):
        root = ET.parse(URDF).getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint")}
        for name, vector in (("knee_L_Joint", self.fn["UPPER"]), ("wheel_L_Joint", self.fn["LOWER"])):
            xyz = [float(v) for v in joints[name].find("origin").attrib["xyz"].split()]
            self.assertAlmostEqual(vector[0], xyz[0])
            self.assertAlmostEqual(vector[1], xyz[2])
        c = self.fn["JumpCfg"]()
        lengths = torch.tensor([c.nominal_length, c.crouch_length, c.thrust_length,
                                c.flight_retract_length, c.prelanding_length, c.landing_absorption_length])
        q = self.fn["leg_pose"](lengths, c.nominal_angle)
        measured, angle = self.fn["leg_kinematics"](q)
        torch.testing.assert_close(measured, lengths[:, None].repeat(1, 2))
        torch.testing.assert_close(angle, torch.full_like(angle, c.nominal_angle))
        for i, name in enumerate(self.fn["LEG_NAMES"]):
            limit = joints[name].find("limit").attrib
            self.assertTrue(bool((q[:, i] >= float(limit["lower"])).all()))
            self.assertTrue(bool((q[:, i] <= float(limit["upper"])).all()))
            self.assertEqual(float(limit["effort"]), 3.)

    def test_fk_matches_independent_urdf_matrix_chain(self):
        # Independent Rodrigues chain includes abduction and lateral offsets.
        root = ET.parse(URDF).getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint")}
        q = torch.tensor([[.2, -.2, -.4, .4, -1., 1.]], dtype=torch.float64)
        lengths, _ = self.fn["leg_kinematics"](q)
        for side, col in (("L", 0), ("R", 1)):
            t = torch.eye(4, dtype=q.dtype)
            origin = None
            for prefix, angle in (("abad", q[0, col]), ("hip", q[0, 2+col]),
                                  ("knee", q[0, 4+col]), ("wheel", 0.)):
                j = joints[f"{prefix}_{side}_Joint"]
                v = torch.tensor([float(x) for x in j.find("origin").attrib["xyz"].split()], dtype=q.dtype)
                axis = torch.tensor([float(x) for x in j.find("axis").attrib["xyz"].split()], dtype=q.dtype)
                x, y, z = axis
                skew = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=q.dtype)
                step = torch.eye(4, dtype=q.dtype)
                step[:3, 3] = v
                step[:3, :3] += math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)
                t = t@step
                if prefix == "hip":
                    origin = t[:3, 3].clone()
            # URDF has a net lateral offset of 0.0251 m below the hip.
            sagittal = torch.sqrt((t[:3, 3]-origin).square().sum() - .0251**2)
            self.assertAlmostEqual(float(sagittal), float(lengths[0, col]), places=6)

    def test_two_wheels_and_clearance_required(self):
        env = make_env(self.fn, 3)
        env.scene["robot"].data.body_pos_w[:, :, 2] = torch.tensor([[.25, .25], [.25, .25], [.25, .04]])
        env.scene["contact_forces"].data.net_forces_w.zero_()
        env.scene["contact_forces"].data.net_forces_w[1, 0, 2] = 3.
        cfg, sensor = SimpleNamespace(name="robot", body_ids=[0, 1]), SimpleNamespace(name="contact_forces", body_ids=[0, 1])
        height, flying, _ = self.fn["jump_measurements"](env, cfg, sensor, .04)
        torch.testing.assert_close(height, torch.tensor([.21, .21, 0.]))
        self.assertEqual(flying.tolist(), [True, False, False])

    def test_full_cycle_success_event_and_partial_reset(self):
        env = make_env(self.fn)
        s = self.fn["get_state"](env)
        r = env.scene["robot"].data
        s.enabled[:] = True
        s.delay[:] = .02
        s.update()
        self.assertEqual(s.phase.tolist(), [1, 1])
        r.joint_pos[:] = self.fn["leg_pose"](torch.full((2,), .115), s.cfg.nominal_angle)
        for _ in range(13):
            s.update()
        self.assertEqual(s.phase.tolist(), [2, 2])
        r.joint_pos[:] = self.fn["leg_pose"](torch.full((2,), .2), s.cfg.nominal_angle)
        r.root_lin_vel_w[:, 2] = 1.
        for _ in range(11):
            s.update()
        self.assertEqual(s.phase.tolist(), [3, 3])
        env.scene["contact_forces"].data.net_forces_w.zero_()
        r.body_pos_w[:, :, 2] = .22
        r.root_pos_w[:, 2] = .4
        s.update()
        self.assertFalse(bool(s.takeoff_event.any()))
        s.update()
        self.assertTrue(bool(s.takeoff_event.all()))
        for _ in range(8):
            s.update()
        self.assertFalse(bool(s.takeoff_event.any()))
        self.assertTrue(bool((s.peak >= .17).all()))
        env.scene["contact_forces"].data.net_forces_w[:] = 5
        r.body_pos_w[:, :, 2] = .04
        r.root_lin_vel_w[:, 2] = -.7
        s.update()
        r.root_lin_vel_w[:, 2] = -.1
        s.update()
        torch.testing.assert_close(s.landing_vz, torch.full((2,), -.7))
        self.assertTrue(bool(s.landing_event.all()))
        r.root_lin_vel_w.zero_()
        for _ in range(45):
            s.update()
        self.assertEqual(s.successes.tolist(), [1., 1.])
        self.assertFalse(bool(s.success_event.any()))
        s.reset(torch.tensor([0]))
        self.assertEqual(s.successes.tolist(), [0., 1.])
        self.assertEqual(s.attempts.tolist(), [0., 1.])

    def test_crouch_failure_is_single_event_not_reset(self):
        env = make_env(self.fn)
        s = self.fn["get_state"](env)
        s.enabled[:] = True
        s.delay[:] = .02
        for _ in range(27):
            s.update()
        self.assertEqual(s.fail_crouch.tolist(), [1., 1.])
        self.assertEqual(s.phase.tolist(), [5, 5])
        for _ in range(30):
            s.update()
        self.assertEqual(s.fail_performance.tolist(), [0., 0.])
        self.assertEqual(s.phase.tolist(), [0, 0])

    def test_strict_target_and_high_reward_uncapped(self):
        cfg = self.fn["JumpCfg"]()
        yes = torch.ones(3, dtype=torch.bool)
        success = self.fn["success_mask"](yes, yes, torch.tensor([.169, .17, .3]),
            torch.ones(3), torch.ones(3), torch.tensor([0., 0., -1.2]), cfg)
        self.assertEqual(success.tolist(), [False, True, False])
        env = make_env(self.fn)
        s = self.fn["get_state"](env)
        s.phase[:] = self.fn["FLIGHT"]
        s.had_flight[:] = True
        s.airborne[:] = True
        s.clearance[:] = torch.tensor([.17, .34])
        reward = self.fn["jump_reward"]
        small = reward(env, "wheel_clearance")
        self.assertLess(float(small[1]), float(small[0]))
        s.cfg.big_jump = True
        torch.testing.assert_close(reward(env, "wheel_clearance"), torch.tensor([1., 2.]))
        s.airborne[:] = False
        self.assertEqual(reward(env, "wheel_clearance").tolist(), [0., 0.])

    def test_no_false_flight_or_success_on_unload_failure(self):
        env = make_env(self.fn)
        s = self.fn["get_state"](env)
        s.phase[:] = self.fn["FLIGHT"]
        s.update()
        self.assertFalse(bool(s.had_flight.any()))
        for _ in range(9):
            s.update()
        self.assertEqual(s.fail_unload.tolist(), [1., 1.])
        self.assertEqual(s.phase.tolist(), [5, 5])
        # Recovery wheel lift is not flight performance.
        env.scene["robot"].data.body_pos_w[:, :, 2] = .5
        env.scene["contact_forces"].data.net_forces_w.zero_()
        for _ in range(105):
            s.update()
        self.assertEqual(s.episode_peak.tolist(), [0., 0.])
        self.assertEqual(s.successes.tolist(), [0., 0.])
        self.assertEqual(s.fail_recovery.tolist(), [1., 1.])

    def test_all_reward_formulas_are_finite_across_phases(self):
        env = make_env(self.fn, 6)
        s = self.fn["get_state"](env)
        s.phase[:] = torch.arange(6)
        kinds = ("lin_vel_z base_height action_smooth nominal_state track_lin_vel track_ang_vel "
                 "track_heading crouch phase_action thrust_pose thrust_speed takeoff takeoff_event "
                 "height wheel_clearance airborne symmetry landing_pose landing_soft landing_impact "
                 "recovery success failure").split()
        for big in (False, True):
            s.cfg.big_jump = big
            for kind in kinds:
                value = self.fn["jump_reward"](env, kind)
                self.assertEqual(value.shape, (6,))
                self.assertTrue(bool(torch.isfinite(value).all()), kind)

    def test_checkpoint_preserves_old_columns_and_zeros_new_features(self):
        spec = importlib.util.spec_from_file_location("motor35_expand", EXPAND)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actor = torch.arange(34.).repeat(2, 1)
        critic = torch.arange(214.).repeat(2, 1)
        history = torch.arange(280.).repeat(2, 1)
        checkpoint = {"model_state_dict": {"actor.0.weight": actor, "actor.6.weight": torch.zeros(8, 128),
                       "critic.0.weight": critic}, "encoder_state_dict": {"encoder.0.weight": history},
                       "optimizer_state_dict": {}, "encoder_optimizer_state_dict": {}, "iter": 10000}
        result = module.expand_checkpoint(checkpoint)
        a = result["model_state_dict"]["actor.0.weight"]
        torch.testing.assert_close(a[:, :31], actor[:, :31])
        torch.testing.assert_close(a[:, -3:], actor[:, -3:])
        torch.testing.assert_close(a[:, 31:34], torch.zeros(2, 3))
        c = result["model_state_dict"]["critic.0.weight"]
        torch.testing.assert_close(c[:, :211], critic[:, :211])
        torch.testing.assert_close(c[:, -3:], critic[:, -3:])
        h = result["encoder_state_dict"]["encoder.0.weight"].reshape(2, 10, 31)
        torch.testing.assert_close(h[:, :, :28], history.reshape(2, 10, 28))
        torch.testing.assert_close(h[:, :, 28:], torch.zeros(2, 10, 3))
        self.assertEqual(result["iter"], 0)
        self.assertIsNone(result["optimizer_state_dict"])


if __name__ == "__main__":
    unittest.main()
