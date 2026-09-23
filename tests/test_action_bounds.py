"""Check real bound methods without constructing the simulator."""
import ast
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def load_method(path, class_name, name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == class_name)
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


loss = load_method("rsl_rl/rsl_rl/algorithm/ppo.py", "PPO", "action_bound_loss")
resolve = load_method("exts/bipedal_locomotion/bipedal_locomotion/utils/wrappers/rsl_rl/vecenv_wrapper.py",
                      "RslRlVecEnvWrapper", "get_action_mean_bounds")


class TestActionBounds(unittest.TestCase):
    def test_gradient_brings_saturated_mean_back_without_touching_legal_mean(self):
        state = SimpleNamespace(action_mean_bounds=torch.tensor([[-1., 1.]] * 3))
        mean = torch.tensor([[-25., 0., 23.]], requires_grad=True)
        loss(state, mean).backward()
        self.assertLess(mean.grad[0, 0], 0)
        self.assertEqual(mean.grad[0, 1], 0)
        self.assertGreater(mean.grad[0, 2], 0)
        self.assertEqual(float(loss(state, torch.zeros(1, 3))), 0.)
        self.assertEqual(float(loss(SimpleNamespace(action_mean_bounds=None), mean)), 0.)

    def test_action_bounds_account_for_offset_scale_and_actual_joint_order(self):
        position = SimpleNamespace(_clip=torch.tensor([[[-2., 2.], [-3., 3.]]]),
                                   _scale=2., _offset=torch.tensor([[0.5, -0.5]]),
                                   _joint_ids=[1, 0],
                                   _asset=SimpleNamespace(data=SimpleNamespace(
                                       joint_pos_limits=torch.tensor([[[-1., 1.], [-0.5, 0.5]]]))))
        velocity = SimpleNamespace(_clip=torch.tensor([[[-15., 15.]]]), _scale=3., _offset=0.)
        terms = {'joint_pos': position, 'joint_vel': velocity}
        env = SimpleNamespace(cfg=SimpleNamespace(getup=True), device='cpu',
                              action_manager=SimpleNamespace(active_terms=list(terms), get_term=terms.get))
        actual = resolve(SimpleNamespace(unwrapped=env))
        torch.testing.assert_close(torch.tensor(actual), torch.tensor([[-0.5, 0.], [-0.25, 0.75], [-5., 5.]]))


class TestFlatTrackingRewards(unittest.TestCase):
    def setUp(self):
        path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py"
        tree = ast.parse(path.read_text())
        names = {"straight_motion_gate", "joint_mirror_pose_l2", "body_y_velocity_error",
                 "body_y_reverse_motion", "straight_feet_alignment", "same_feet_x_position"}
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = {"torch": torch, "SceneEntityCfg": lambda name: SimpleNamespace(name=name)}
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *funcs], type_ignores=[])), str(path), "exec"), scope)
        self.fn = scope
        self.command = torch.tensor([[0., 2., 0.], [0., -2., 0.], [0., 0., 0.]])
        self.data = SimpleNamespace(joint_pos=torch.tensor([[.1, -.1, .4, -.4, -.5, .5]] * 3),
                                    root_lin_vel_b=torch.zeros(3, 3), root_ang_vel_b=torch.zeros(3, 3))
        self.env = SimpleNamespace(scene={"robot": SimpleNamespace(data=self.data)},
                                   command_manager=SimpleNamespace(get_command=lambda _: self.command))
        self.cfg = SimpleNamespace(name="robot", joint_ids=[0, 1, 2, 3, 4, 5])

    def test_large_error_still_distinguishes_progress(self):
        values = []
        for speed in [-.5, 0., .5, 1., 2.]:
            self.data.root_lin_vel_b[:, 1] = torch.tensor([speed, -speed, 0.])
            value = self.fn["body_y_velocity_error"](self.env, "base_velocity", self.cfg)
            self.assertAlmostEqual(float(value[0]), float(value[1]))
            self.assertEqual(float(value[2]), 0.)
            values.append(float(value[0]))
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_reverse_penalty_is_bidirectional_and_off_at_standstill(self):
        self.data.root_lin_vel_b[:, 1] = torch.tensor([-.5, .5, 2.])
        value = self.fn["body_y_reverse_motion"](self.env, "base_velocity", self.cfg)
        torch.testing.assert_close(value, torch.tensor([.5, .5, 0.]))
        self.data.root_lin_vel_b[:, 1] *= -1
        torch.testing.assert_close(self.fn["body_y_reverse_motion"](self.env, "base_velocity", self.cfg),
                                   torch.zeros(3))

    def test_pose_mirror_and_turn_gate(self):
        f = self.fn["joint_mirror_pose_l2"]
        torch.testing.assert_close(f(self.env, "base_velocity", self.cfg), torch.zeros(3))
        self.data.joint_pos[:, 0] = .3
        self.command[:, 2] = torch.tensor([0., .25, .5])
        value = f(self.env, "base_velocity", self.cfg)
        self.assertGreater(float(value[0]), 0.)
        self.assertAlmostEqual(float(value[1]), float(value[0]) * .5)
        self.assertEqual(float(value[2]), 0.)
        self.data.root_ang_vel_b[:, 2] = -.5
        torch.testing.assert_close(f(self.env, "base_velocity", self.cfg), torch.zeros(3))

    def test_config_order_and_high_speed_reward_budget(self):
        path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_wheelfoot_env_cfg.py"
        tree = ast.parse(path.read_text())
        f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure_flat_tracking_rewards")
        mdp = SimpleNamespace(**{name: self.fn[name] for name in ("joint_mirror_pose_l2", "body_y_velocity_error",
                                                                "body_y_reverse_motion", "straight_feet_alignment")})
        scope = {"mdp": mdp, "RewTerm": SimpleNamespace,
                 "SceneEntityCfg": lambda name, **kw: SimpleNamespace(name=name, **kw)}
        exec(compile(ast.Module(body=[f], type_ignores=[]), str(path), "exec"), scope)
        cfg = SimpleNamespace(rewards=SimpleNamespace(pen_joint_vel_wheel_l2=SimpleNamespace(weight=-.0005)))
        scope["configure_flat_tracking_rewards"](cfg)
        spec = cfg.rewards.rew_leg_symmetry.params["asset_cfg"]
        self.assertTrue(spec.preserve_order)
        self.assertEqual(spec.joint_names, [f"{kind}_{side}_Joint" for kind in ("abad", "hip", "knee") for side in ("L", "R")])
        weight = cfg.rewards.pen_joint_vel_wheel_l2.weight
        self.assertLess(abs(weight) * 2 * (2 / .0375)**2, .3)
        def total(v):
            self.data.root_lin_vel_b[0, 1] = v
            error = float(self.fn["body_y_velocity_error"](self.env, "base_velocity", self.cfg)[0])
            reverse = float(self.fn["body_y_reverse_motion"](self.env, "base_velocity", self.cfg)[0])
            return (3 * math.exp(-(2-v)**2/.2) + weight*2*(v/.0375)**2
                    + cfg.rewards.pen_body_y_velocity_error.weight*error
                    + cfg.rewards.pen_reverse_motion.weight*reverse)
        values = [total(v) for v in [-.5, 0., .5, 1., 1.5, 2.]]
        self.assertTrue(all(a < b for a,b in zip(values,values[1:])))


class TestAbadTorquePenalty(unittest.TestCase):
    def test_excess_penalty_at_rest_selects_only_abad(self):
        path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "joint_torque_excess_l1")
        scope = {"torch": torch, "ManagerBasedRLEnv": object, "SceneEntityCfg": object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
        data = SimpleNamespace(
            applied_torque=torch.tensor([[100., .5, -.5, 100.], [100., 1., -1.5, 100.]]),
            joint_vel=torch.zeros(2, 4),
        )
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
        cfg = SimpleNamespace(name="robot", joint_ids=[1, 2])
        actual = scope["joint_torque_excess_l1"](env, cfg, .5)
        torch.testing.assert_close(actual, torch.tensor([0., 1.5]))


class TestFlatBaseline(unittest.TestCase):
    """Test the actual configuration helper without starting Kit or allocating GPU memory."""

    def setUp(self):
        path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_wheelfoot_env_cfg.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "configure_flat_baseline")
        self.mdp = SimpleNamespace(reset_scene_to_default=object(), base_height_l2=object())
        scope = {"mdp": self.mdp, "EventTerm": SimpleNamespace,
                 "SceneEntityCfg": lambda name, **kw: SimpleNamespace(name=name, **kw)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
        self.removed = ("add_base_mass", "add_link_mass", "radomize_rigid_body_mass_inertia",
                        "robot_joint_stiffness_and_damping", "robot_center_of_mass",
                        "randomize_actuator_gains", "push_robot", "reset_robot_joints")
        self.cfg = SimpleNamespace(
            events=SimpleNamespace(**{name: object() for name in self.removed},
                robot_physics_material=SimpleNamespace(params={}),
                wheel_physics_material=SimpleNamespace(params={})),
            observations=SimpleNamespace(**{name: SimpleNamespace(enable_corruption=True)
                for name in ("policy", "critic", "commands", "obsHistory")}),
            scene=SimpleNamespace(robot=SimpleNamespace(init_state=SimpleNamespace()),
                terrain=SimpleNamespace(physics_material=SimpleNamespace(restitution=1.0))),
            rewards=SimpleNamespace(pen_base_height=SimpleNamespace(func=None, params={}),
                                    pen_joint_vel_wheel_l2=SimpleNamespace(weight=-0.005)),
            commands=object(), terminations=object(),
        )
        self.configure = scope["configure_flat_baseline"]

    def test_randomization_off_but_reset_and_task_commands_preserved(self):
        commands, terminations = self.cfg.commands, self.cfg.terminations
        self.configure(self.cfg)
        for name in self.removed:
            self.assertIsNone(getattr(self.cfg.events, name))
        for group in vars(self.cfg.observations).values():
            self.assertFalse(group.enable_corruption)
        for name in ("robot_physics_material", "wheel_physics_material"):
            params = getattr(self.cfg.events, name).params
            for key in ("static_friction_range", "dynamic_friction_range", "restitution_range"):
                self.assertEqual(params[key][0], params[key][1])
            self.assertEqual(params["num_buckets"], 1)
            self.assertLessEqual(params["dynamic_friction_range"][0], params["static_friction_range"][0])
        reset = self.cfg.events.reset_robot_base
        self.assertIs(reset.func, self.mdp.reset_scene_to_default)
        self.assertEqual(reset.mode, "reset")
        self.assertTrue(reset.params["reset_joint_targets"])
        self.assertIs(commands, self.cfg.commands)
        self.assertIs(terminations, self.cfg.terminations)
        self.assertEqual(self.cfg.scene.terrain.physics_material.restitution, 0.0)

    def test_velocity_tracking_outweighs_deliberate_underspeed(self):
        self.configure(self.cfg)
        weight = self.cfg.rewards.pen_joint_vel_wheel_l2.weight
        def reward(speed):
            return 3.0 * math.exp(-(speed - 0.3) ** 2 / 0.2) + weight * 2 * (speed / 0.0375) ** 2
        self.assertGreater(reward(0.3), reward(0.2))
        self.assertIs(self.cfg.rewards.pen_base_height.func, self.mdp.base_height_l2)

    def test_nominal_pose_limits_ground_clearance_and_com(self):
        self.configure(self.cfg)
        initial = self.cfg.scene.robot.init_state
        path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/WF_TRON1A.urdf"
        robot = ET.parse(path).getroot()
        transforms = {"base_Link": torch.eye(4, dtype=torch.float64)}
        transforms["base_Link"][:3, 3] = torch.tensor(initial.pos, dtype=torch.float64)
        for joint in robot.findall("joint"):
            self.assertEqual(joint.find("origin").get("rpy"), "0 0 0")
            angle = initial.joint_pos[joint.get("name")]
            if joint.get("type") != "continuous":
                limit = joint.find("limit")
                self.assertGreater(angle, float(limit.get("lower")))
                self.assertLess(angle, float(limit.get("upper")))
            axis = [float(x) for x in joint.find("axis").get("xyz").split()]
            x, y, z = axis
            cross = torch.tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=torch.float64)
            transform = torch.eye(4, dtype=torch.float64)
            transform[:3, :3] += math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)
            transform[:3, 3] = torch.tensor([float(x) for x in joint.find("origin").get("xyz").split()])
            transforms[joint.find("child").get("link")] = transforms[joint.find("parent").get("link")] @ transform
        wheel_y = []
        for side in ("L", "R"):
            pose = transforms[f"wheel_{side}_Link"]
            clearance = float(pose[2, 3]) - 0.0375
            self.assertGreater(clearance, 0.0005)
            self.assertLess(clearance, 0.0035)
            wheel_y.append(float(pose[1, 3]))
        total_mass, com_y = 0., 0.
        for link in robot.findall("link"):
            mass = float(link.find("inertial/mass").get("value"))
            local = torch.tensor([float(x) for x in link.find("inertial/origin").get("xyz").split()] + [1.],
                                 dtype=torch.float64)
            com_y += mass * float((transforms[link.get("name")] @ local)[1])
            total_mass += mass
        self.assertAlmostEqual(com_y / total_mass, sum(wheel_y) / 2, delta=0.003)

    def test_wheel_collision_is_centered_on_visual_mesh(self):
        path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/WF_TRON1A.urdf"
        links = {link.get("name"): link for link in ET.parse(path).getroot().findall("link")}
        for side in ("L", "R"):
            link = links[f"wheel_{side}_Link"]
            origin = link.find("collision/origin")
            center_x, center_y, center_z = map(float, origin.get("xyz").split())
            mesh_path = path.parent / link.find("visual/geometry/mesh").get("filename")
            mesh_x_bounds = trimesh.load_mesh(mesh_path).bounds[:, 0]
            self.assertAlmostEqual(center_x, float(mesh_x_bounds.mean()), places=6)
            self.assertEqual((center_y, center_z), (0.0, 0.0))
            self.assertAlmostEqual(float(link.find("collision/geometry/cylinder").get("length")),
                                   float(mesh_x_bounds[1] - mesh_x_bounds[0]), places=6)

    def test_base_collision_keeps_upper_ellipsoid_and_clips_lower_to_visual_hull(self):
        path = ROOT / "exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/WF_TRON1A.urdf"
        base = next(link for link in ET.parse(path).getroot().findall("link")
                    if link.get("name") == "base_Link")
        self.assertEqual(base.find("visual/geometry/mesh").get("filename"), "meshes/base_link.STL")
        self.assertEqual(float(base.find("inertial/mass").get("value")), 2.4)
        from tools.prepare_wheel_leg_urdf import BASE_ELLIPSOID_CENTER, BASE_ELLIPSOID_RADII
        import numpy as np
        from scipy.spatial import ConvexHull
        center, radii = np.asarray(BASE_ELLIPSOID_CENTER), np.asarray(BASE_ELLIPSOID_RADII)
        visual = trimesh.load_mesh(path.parent / "meshes/base_link.STL")
        self.assertLessEqual(np.linalg.norm((visual.vertices - center) / radii, axis=1).max(), 1)
        collisions = base.findall("collision")
        self.assertEqual(len(collisions), 2)
        self.assertEqual(len({collision.get("name") for collision in collisions}), len(collisions))
        meshes = []
        for collision in collisions:
            origin = collision.find("origin")
            self.assertEqual(origin.get("xyz"), "0 0 0")
            self.assertEqual(origin.get("rpy"), "0 0 0")
            filename = collision.find("geometry/mesh").get("filename")
            mesh = trimesh.load_mesh(path.parent / filename)
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_convex)
            self.assertLessEqual(np.linalg.norm((mesh.vertices - center) / radii, axis=1).max(), 1.001)
            meshes.append(mesh)
        upper, lower = meshes
        self.assertGreaterEqual(upper.vertices[:, 2].min(), center[2] - 1e-7)
        self.assertLessEqual(lower.vertices[:, 2].max(), center[2] + 1e-7)
        self.assertAlmostEqual(upper.bounds[1, 2], center[2] + radii[2], places=6)
        expected_upper_volume = 2 * np.pi * np.prod(radii) / 3
        self.assertAlmostEqual(upper.volume / expected_upper_volume, 1.0, delta=0.002)
        self.assertGreaterEqual(len(upper.faces), 10000)
        self.assertGreater((upper.volume + lower.volume) / (2 * expected_upper_volume), 0.7)
        hull_planes = ConvexHull(visual.vertices).equations
        self.assertLessEqual(np.max(lower.vertices @ hull_planes[:, :3].T
                                    + hull_planes[:, 3]), 1e-6)

        from tools.prepare_wheel_leg_urdf import prepare
        source = ROOT.parent / "轮腿总装111/轮腿总装111.urdf"
        with tempfile.TemporaryDirectory() as directory:
            regenerated = Path(directory) / "WF_TRON1A.urdf"
            prepare(source, regenerated)
            self.assertEqual(ET.tostring(ET.parse(regenerated).getroot()),
                             ET.tostring(ET.parse(path).getroot()))
        joints = {joint.get("name"): joint for joint in ET.parse(path).getroot().findall("joint")}
        self.assertEqual(float(joints["hip_L_Joint"].find("limit").get("upper")), 0.0)
        self.assertEqual(float(joints["hip_R_Joint"].find("limit").get("lower")), 0.0)
        for collision in collisions:
            self.assertTrue((path.parent / collision.find("geometry/mesh").get("filename")).exists())


if __name__ == '__main__':
    unittest.main()
