"""Validate nonzero-pose FK and conservative ground placement without Isaac Sim."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'exts/bipedal_locomotion/bipedal_locomotion/tasks/recovery/mdp/nominal_geometry.py'
spec = importlib.util.spec_from_file_location('nominal_geometry', MODULE)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


class TestNominalGeometry(unittest.TestCase):
    def test_rotated_joint_moves_child_collision_before_root_tilt(self):
        urdf = '''<robot name="fixture"><link name="base_Link"/><link name="child">
        <collision><origin xyz="1 0 0"/><geometry><box size=".2 .2 .2"/></geometry></collision>
        </link><joint name="j" type="revolute"><parent link="base_Link"/><child link="child"/>
        <origin xyz="0 0 1"/><axis xyz="0 1 0"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.urdf'
            path.write_text(urdf)
            points = geometry.nominal_collision_corners(path, {'j': np.pi / 2})
        np.testing.assert_allclose(points.mean(0), np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(points.min(0), [-.1] * 3, atol=1e-6)

    def test_adapted_robot_clearance_at_all_curriculum_orientations(self):
        positions = {f'{kind}_{side}_Joint': 0. for kind in ('abad', 'hip', 'knee', 'wheel') for side in ('L', 'R')}
        positions.update(hip_L_Joint=.13437526, hip_R_Joint=-.13437526,
                         knee_L_Joint=-.53190465, knee_R_Joint=.53190465)
        points = geometry.nominal_collision_corners(
            ROOT / 'exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/WF_TRON1A.urdf', positions)
        self.assertTrue(np.isfinite(points).all())
        self.assertAlmostEqual(-points[:, 2].min(), .18, delta=.005)
        heights = []
        for degrees in range(0, 181, 5):
            for axis in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
                rotation = trimesh.transformations.rotation_matrix(np.deg2rad(degrees), axis)
                rotated = trimesh.transform_points(points, rotation)
                height = -rotated[:, 2].min() + .005
                self.assertAlmostEqual(float((rotated[:, 2] + height).min()), .005, places=6)
                heights.append(height)
        self.assertLess(min(heights), .12)  # an old 12 cm floor would suspend some fallen poses
