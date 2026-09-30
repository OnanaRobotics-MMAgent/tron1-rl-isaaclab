import copy
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('fallen_controller',Path(__file__).resolve().parents[2]/'scripts/recovery/continuous_fallen.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class TestFallenController(unittest.TestCase):
    def test_mastery_needs_every_stage_direction_and_no_violation(self):
        stages=[dict(episodes=128,success_rate=.95,direction_stats=[dict(episodes=32,success_rate=.90) for _ in range(4)]) for _ in range(3)]
        result=dict(stages=stages,success_rate=.95,leg_limit_violations=0)
        self.assertTrue(m.mastered(result))
        bad=copy.deepcopy(result);bad['stages'][2]['direction_stats'][1]['success_rate']=.79
        self.assertFalse(m.mastered(bad))
        bad=copy.deepcopy(result);bad['stages'][2]['direction_stats'][1]['episodes']=2
        self.assertFalse(m.mastered(bad))
        bad=copy.deepcopy(result);bad['leg_limit_violations']=1
        self.assertFalse(m.mastered(bad))
