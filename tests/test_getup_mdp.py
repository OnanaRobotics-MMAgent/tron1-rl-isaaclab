"""CPU tests for recovery semantics; no Kit launch or Isaac Lab mocks required."""

import importlib
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch

# Load only the simulator-independent terms, bypassing the extension's Gym registration.
package = ModuleType("getup_terms_under_test")
package.__path__ = [str(Path(__file__).resolve().parents[1] /
                       "exts/bipedal_locomotion/bipedal_locomotion/tasks/recovery/mdp")]
sys.modules[package.__name__] = package
rewards = importlib.import_module(package.__name__ + ".rewards")
terminations = importlib.import_module(package.__name__ + ".terminations")
curriculum = importlib.import_module(package.__name__ + ".curriculum")
get_state = importlib.import_module(package.__name__ + ".state").get_state
limits = importlib.import_module(package.__name__ + ".limits")


class Scene(dict):
    pass


def make_env(n=4):
    # Current WF_TRON1A URDF: the neutral wheel bottom is about 0.1508 m below
    # base_Link, so the nominal root height is 0.18 m (with reset clearance).
    cfg = SimpleNamespace(initial_level=0, curriculum_enabled=True, min_curriculum_episodes=4,
                          promote_success_rate=0.75, target_height=0.18, success_tilt=math.radians(15),
                          height_tolerance=0.07, max_linear_speed=0.25, max_angular_speed=0.5,
                          min_wheel_force=5.0, max_body_force=5.0, max_pose_error=0.16,
                          min_episode_time=0.0, hold_time=0.06, max_drift=2.5, joint_limit_tolerance=0.05)
    data = SimpleNamespace(projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
                           root_pos_w=torch.tensor([[0., 0., 0.18]]).repeat(n, 1),
                           root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3),
                           joint_pos=torch.zeros(n, 8), default_joint_pos=torch.zeros(n, 8),
                           joint_pos_limits=torch.tensor([-1.0, 1.0]).expand(n, 8, 2).clone())
    forces = torch.zeros(n, 3, 3)
    forces[:, :2, 2] = 40.0
    scene = Scene(robot=SimpleNamespace(data=data), contact_forces=SimpleNamespace(
        data=SimpleNamespace(net_forces_w=forces)))
    scene.env_origins = torch.zeros(n, 3)
    return SimpleNamespace(scene=scene, cfg=SimpleNamespace(getup=cfg), device="cpu", num_envs=n,
                           step_dt=0.02, common_step_counter=0)


ENTITIES = dict(wheel_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 1]),
                body_cfg=SimpleNamespace(name="contact_forces", body_ids=[2]),
                leg_cfg=SimpleNamespace(name="robot", joint_ids=list(range(6))))


class TestGetUp(unittest.TestCase):
    def test_auto_curriculum_requires_all_directions_and_consecutive_windows(self):
        env = make_env(8)
        env.cfg.getup.tilt_ranges_deg = tuple((i, i + 5) for i in range(10))
        env.cfg.getup.min_curriculum_episodes = 8
        env.cfg.getup.promote_windows = 3
        env.cfg.getup.promote_direction_success_rate = 0.7
        env.cfg.getup.min_direction_episodes = 2
        state = get_state(env)
        state.level = 4
        state.stage[:] = 4
        state.direction[:] = torch.arange(8) % 4
        state.active[:] = True
        state.steps[:] = 100
        ids = torch.arange(8)
        state.success[:] = True
        curriculum.recovery_levels(env, ids)
        self.assertEqual(state.promotion_streak, 1)
        # 75% overall meets this fixture's threshold, but one direction fails.
        state.success[state.direction == 0] = False
        curriculum.recovery_levels(env, ids)
        self.assertEqual(state.promotion_streak, 0)
        self.assertEqual(state.level, 4)
        state.success[:] = True
        for _ in range(2):
            curriculum.recovery_levels(env, ids)
            self.assertEqual(state.level, 4)
        saved = state.state_dict()
        state.load_state_dict(saved)
        curriculum.recovery_levels(env, ids)
        self.assertEqual(state.level, 5)
        self.assertEqual(state.promotion_streak, 0)
        self.assertEqual(state.metric_windows[-1]['GetUp/stage_04/success_rate'], 1.)
        self.assertNotIn('GetUp/stage_05/success_rate', state.metric_windows[-1])
        curriculum.recovery_levels(env, ids)
        self.assertEqual(state.window_episodes, 0)  # old-level episodes excluded

    def test_low_height_penalty_has_grace_and_disappears_at_target(self):
        env = make_env()
        state = get_state(env)
        env.scene['robot'].data.root_pos_w[:, 2] = torch.tensor([0.04, 0.1215, 0.18, 0.25])
        torch.testing.assert_close(rewards.low_height_deficit(env), torch.zeros(4))
        state.steps[:] = 51
        torch.testing.assert_close(rewards.low_height_deficit(env), torch.tensor([1., 0.5, 0., 0.]))

    def test_upside_down_is_not_upright(self):
        env = make_env()
        env.scene["robot"].data.projected_gravity_b[0, 2] = 1.
        self.assertEqual(rewards.upright(env).tolist(), [0., 1., 1., 1.])
        self.assertFalse(bool(rewards.stable_mask(env, **ENTITIES)[0]))

    def test_success_requires_both_wheels_no_body_support_and_low_speed(self):
        env = make_env()
        env.scene["contact_forces"].data.net_forces_w[0, 0, 2] = 0.
        env.scene["contact_forces"].data.net_forces_w[1, 2, 2] = 10.
        env.scene["robot"].data.root_lin_vel_w[2, 0] = 0.5
        self.assertEqual(rewards.stable_mask(env, **ENTITIES).tolist(), [False, False, False, True])

    def test_hold_is_consecutive_and_updated_once_per_step(self):
        env = make_env()
        state = get_state(env)
        for step in (1, 2):
            env.common_step_counter = step
            self.assertFalse(bool(terminations.sustained_success(env, **ENTITIES).any()))
            terminations.sustained_success(env, **ENTITIES)
            self.assertEqual(int(state.hold[0]), step)
        env.scene["robot"].data.root_ang_vel_w[0, 0] = 1.0
        env.common_step_counter = 3
        self.assertEqual(terminations.sustained_success(env, **ENTITIES).tolist(), [False, True, True, True])
        self.assertEqual(int(state.hold[0]), 0)

    def test_height_is_local_to_environment(self):
        env = make_env()
        env.scene.env_origins[:, 2] = 10.
        env.scene["robot"].data.root_pos_w[:, 2] += 10.
        self.assertTrue(bool(rewards.stable_mask(env, **ENTITIES).all()))
        torch.testing.assert_close(rewards.height_tracking(env), torch.ones(4))

    def test_recovery_height_signal_is_dense_and_capped(self):
        env = make_env()
        env.scene["robot"].data.root_pos_w[:, 2] = torch.tensor([0.05, 0.1215, 0.18, 0.30])
        value = rewards.height_progress(env)
        torch.testing.assert_close(value, torch.tensor([0.0, 0.5, 1.0, 1.0]))

    def test_inference_step_allows_external_reset(self):
        env = make_env()
        state = get_state(env)
        with torch.inference_mode():
            env.common_step_counter = 1
            terminations.sustained_success(env, **ENTITIES)
        # Diagnostic tools and Gym users may reset outside inference_mode.
        state.hold.zero_()
        state.success.zero_()
        self.assertFalse(bool(state.hold.any()))

    def test_standing_outside_area_does_not_count_as_success(self):
        env = make_env()
        env.scene["robot"].data.root_pos_w[0, 0] = 3.0
        for step in range(1, 5):
            env.common_step_counter = step
            result = terminations.sustained_success(env, **ENTITIES)
        self.assertFalse(bool(result[0]))
        self.assertTrue(bool(terminations.outside_recovery_area(env)[0]))

    def test_success_bonus_is_step_size_independent(self):
        env = make_env()
        get_state(env).success[0] = True
        for dt in (0.01, 0.02, 0.04):
            env.step_dt = dt
            self.assertAlmostEqual(float(rewards.success_bonus(env)[0]) * dt, 1.)

    def test_curriculum_ignores_initial_reset_and_replay(self):
        env = make_env()
        state = get_state(env)
        ids = torch.arange(4)
        curriculum.recovery_levels(env, ids)
        self.assertEqual(state.window_episodes, 0)
        state.active[:] = True
        state.steps[:] = 100
        state.success[:] = True
        curriculum.recovery_levels(env, ids)
        self.assertEqual(state.level, 1)
        # Old-level episodes finishing after promotion must not promote again.
        curriculum.recovery_levels(env, ids)
        self.assertEqual(state.window_episodes, 0)
        state.stage[:] = 1
        state.success[:] = False
        curriculum.recovery_levels(env, ids)
        self.assertEqual(state.level, 1)
        self.assertEqual(state.last_rate, 0.)

    def test_fixed_stage_never_promotes(self):
        env = make_env()
        env.cfg.getup.curriculum_enabled = False
        state = get_state(env)
        state.active[:] = True
        state.steps[:] = 100
        state.success[:] = True
        curriculum.recovery_levels(env, torch.arange(4))
        self.assertEqual(state.level, 0)

    def test_leg_full_turn_is_rejected_without_angle_wrapping(self):
        env = make_env()
        env.scene["robot"].data.joint_pos[0, 0] = 2 * math.pi
        env.scene["robot"].data.joint_pos[1, 6] = 20 * math.pi  # wheel is excluded
        result = terminations.leg_limit_violation(env, ENTITIES["leg_cfg"])
        self.assertEqual(result.tolist(), [True, False, False, False])
        for step in range(1, 5):
            env.common_step_counter = step
            success = terminations.sustained_success(env, **ENTITIES)
        self.assertFalse(bool(success[0]))
        self.assertTrue(bool(success[1]))

    def test_limits_do_not_enlarge_asset_and_reject_continuous_legs(self):
        names = list(limits.LEG_JOINT_LIMITS)
        usd = torch.tensor([-0.25, 0.25]).expand(2, 6, 2).clone()
        torch.testing.assert_close(limits.validated_leg_limits(names, limits.LEG_JOINT_LIMITS, usd), usd)
        invalid = dict(limits.LEG_JOINT_LIMITS)
        invalid[names[0]] = (-2 * math.pi, 2 * math.pi)
        with self.assertRaises(ValueError):
            limits.validated_leg_limits(names, invalid, usd)
        invalid = dict(limits.LEG_JOINT_LIMITS)
        invalid[names[0]] = (0.1, 0.5)
        with self.assertRaises(ValueError):
            limits.validated_leg_limits(names, invalid, usd)

    def test_curriculum_checkpoint_does_not_restore_episodes(self):
        env = make_env()
        state = get_state(env)
        state.level, state.window_episodes, state.window_successes = 2, 13, 8
        state.success[:] = True
        restored = get_state(make_env())
        restored.load_state_dict(state.state_dict())
        self.assertEqual(restored.state_dict(), state.state_dict())
        self.assertFalse(bool(restored.success.any()))
        with self.assertRaises(ValueError):
            restored.load_state_dict(dict(state.state_dict(), level=4))


if __name__ == "__main__":
    unittest.main()
