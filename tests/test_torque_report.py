"""Averages must not hide physical peaks or mix/reset-pad unequal episodes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('torque_report',
    Path(__file__).resolve().parents[1] / 'scripts/rsl_rl/torque_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class TestTorqueReport(unittest.TestCase):
    def test_signed_mean_and_unaveraged_peak_are_distinct(self):
        values = np.array([[20.], [-20.], [4.], [-4.], [3.]])
        np.testing.assert_allclose(report.block_mean(values, 4), [[0.], [3.]])
        stats = report.statistics(values, ['j'])['j']
        self.assertEqual(stats['peak_abs_Nm'], 20.)
        self.assertEqual(stats['min_Nm'], -20.)
        self.assertEqual(stats['max_Nm'], 20.)
        self.assertAlmostEqual(stats['mean_abs_Nm'], 10.2)

    def test_episode_alignment_never_zero_pads_ended_episodes(self):
        mean, counts = report.aligned_mean([np.array([[2.], [4.]]), np.array([[6.]])])
        np.testing.assert_allclose(mean, [[4.], [4.]])
        np.testing.assert_array_equal(counts, [2, 1])

    def test_partial_is_saved_but_excluded_from_cross_episode_average(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(report, 'plot_curve'):
            out = Path(directory) / 'run'
            sink = report.TorqueReport(out, ['j'], .005, 4, {})
            values = np.ones((4, 2, 1))
            sink.save_episode(values, dict(complete=True, outcome='success'))
            sink.save_episode(values * 20, dict(complete=False, outcome='interrupted'))
            sink.finish()
            self.assertEqual(len(sink.curves), 1)
            self.assertTrue((out / 'episode_0001/physics_torques.csv').exists())
            self.assertIn('20.0,1', (out / 'overall_maxima.csv').read_text())
            with self.assertRaises(FileExistsError):
                report.TorqueReport(out, ['j'], .005, 4, {})
