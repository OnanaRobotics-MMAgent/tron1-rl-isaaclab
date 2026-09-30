import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('continuous_getup', Path(__file__).resolve().parents[2] / 'scripts/rsl_rl/continuous_getup.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestContinuousGetUp(unittest.TestCase):
    def test_regression_compares_same_stages_not_new_difficulty(self):
        old = [dict(stage=2, success_rate=.95)]
        new = [dict(stage=2, success_rate=.9), dict(stage=3, success_rate=.3)]
        self.assertEqual(module.regressed_stages(old, new), [])
        new[0]['success_rate'] = .5
        self.assertEqual(module.regressed_stages(old, new), [2])

    def test_completion_requires_all_directions_samples_and_no_violation(self):
        result = dict(success_rate=.9, direction_stats=[dict(episodes=128, success_rate=.8) for _ in range(4)],
                      leg_limit_violations=0)
        self.assertTrue(module.passed([result]))
        self.assertFalse(module.passed([]))
        result['direction_stats'][2]['success_rate'] = .69
        self.assertFalse(module.passed([result]))
        result['direction_stats'][2]['success_rate'] = .8
        result['direction_stats'][2]['episodes'] = 0
        self.assertFalse(module.passed([result]))
        result['direction_stats'][2]['episodes'] = 128
        result['leg_limit_violations'] = 1
        self.assertFalse(module.passed([result]))

    def test_state_replacement_is_complete(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'state.json'
            module.atomic_json(p, dict(iteration=15000, phase='train'))
            module.atomic_json(p, dict(iteration=15001, phase='evaluate'))
            self.assertEqual(json.loads(p.read_text()), dict(iteration=15001, phase='evaluate'))
            self.assertFalse(p.with_suffix('.json.tmp').exists())
