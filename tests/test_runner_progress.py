"""Exercise the real logger without importing the simulator or constructing PPO."""

import ast
import contextlib
import io
import importlib.util
from pathlib import Path
import statistics
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


source = Path(__file__).resolve().parents[1] / "rsl_rl/rsl_rl/runner/on_policy_runner.py"
runner = next(node for node in ast.parse(source.read_text()).body
              if isinstance(node, ast.ClassDef) and node.name == "OnPolicyRunner")
log_method = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "log")
namespace = {"torch": torch, "statistics": statistics}
exec(compile(ast.Module(body=[log_method], type_ignores=[]), str(source), "exec"), namespace)


class TestRunnerProgress(unittest.TestCase):
    def check_progress(self, start, count, prior_time=0.0):
        state = SimpleNamespace(
            num_steps_per_env=24, env=SimpleNamespace(num_envs=4096),
            tot_timesteps=0, tot_time=prior_time, writer=Mock(),
            alg=SimpleNamespace(actor_critic=SimpleNamespace(logstd=torch.zeros(8)), learning_rate=1e-4),
        )
        for offset in range(count):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                namespace["log"](state, dict(
                    collection_time=0.6, learn_time=0.4, ep_infos=[],
                    mean_value_loss=0.0, mean_extra_loss=0.0, mean_surrogate_loss=0.0,
                    mean_kl=0.0, rewbuffer=[], it=start + offset,
                    start_iteration=start, start_total_time=prior_time, tot_iter=start + count,
                ))
            rendered = output.getvalue()
            self.assertIn(f"Learning iteration {start + offset + 1}/{start + count}", rendered)
            eta = rendered.split("ETA:")[1].strip()
            self.assertEqual(eta, f"{count - offset - 1:.1f}s")
        self.assertEqual(state.tot_timesteps, count * 24 * 4096)

    def test_fresh_training(self):
        self.check_progress(0, 3)

    def test_resumed_training(self):
        self.check_progress(6800, 500)

    def test_repeated_learn_excludes_previous_elapsed_time(self):
        self.check_progress(7300, 3, prior_time=600.0)


class TestRunnerConfiguration(unittest.TestCase):
    def test_experiment_name_override_and_default(self):
        import argparse

        path = source.parents[3] / "scripts/rsl_rl/cli_args.py"
        spec = importlib.util.spec_from_file_location("cli_args_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parser = argparse.ArgumentParser()
        module.add_rsl_rl_args(parser)
        for argv, expected in (([], "original"),
                               (["--experiment_name", "wf_smoke"], "wf_smoke")):
            cfg = SimpleNamespace(experiment_name="original", logger="tensorboard")
            module.update_rsl_rl_cfg(cfg, parser.parse_args(argv))
            self.assertEqual(cfg.experiment_name, expected)

    def test_flat_baseline_disables_random_initial_episode_phase(self):
        path = source.parents[3] / "scripts/rsl_rl/train.py"
        tree = ast.parse(path.read_text())
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "learn")
        expr = next(kw.value for kw in call.keywords if kw.arg == "init_at_random_ep_len")
        code = compile(ast.Expression(expr), str(path), "eval")
        for cfg, expected in ((SimpleNamespace(deterministic_baseline=True), False),
                              (SimpleNamespace(getup=object(), deterministic_baseline=False), False),
                              (SimpleNamespace(), True)):
            self.assertEqual(eval(code, {"env_cfg": cfg}), expected)


if __name__ == "__main__":
    unittest.main()
