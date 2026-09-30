"""Advance using completed episodes at the current difficulty, excluding replay."""

import torch

from .state import get_state


def recovery_levels(env, env_ids):
    state = get_state(env)
    cfg = env.cfg.getup
    if env_ids is not None and len(env_ids):
        eligible = state.active[env_ids] & (state.stage[env_ids] == state.level) & (state.steps[env_ids] > 0)
        state.window_episodes += int(eligible.sum())
        state.window_successes += int((eligible & state.success[env_ids]).sum())
        direction_threshold = getattr(cfg, "promote_direction_success_rate", 0.0)
        if direction_threshold > 0:
            counts = torch.bincount(state.direction[env_ids][eligible], minlength=4).tolist()
            successes = torch.bincount(state.direction[env_ids][eligible & state.success[env_ids]], minlength=4).tolist()
            state.direction_episodes = [a + b for a, b in zip(state.direction_episodes, counts)]
            state.direction_successes = [a + b for a, b in zip(state.direction_successes, successes)]
        if state.window_episodes >= cfg.min_curriculum_episodes:
            state.last_rate = state.window_successes / state.window_episodes
            prefix = f"GetUp/stage_{state.level:02d}"
            metrics = {f"{prefix}/success_rate": state.last_rate,
                       f"{prefix}/episodes": state.window_episodes}
            for direction, (n, k) in enumerate(zip(state.direction_episodes, state.direction_successes)):
                if n:
                    metrics[f"{prefix}/direction_{direction}_success_rate"] = k / n
            state.metric_windows.append(metrics)
            directions_pass = direction_threshold <= 0 or all(
                n >= getattr(cfg, "min_direction_episodes", 128) and k / max(n, 1) >= direction_threshold
                for n, k in zip(state.direction_episodes, state.direction_successes))
            passed = state.last_rate >= cfg.promote_success_rate and directions_pass
            state.promotion_streak = state.promotion_streak + 1 if passed else 0
            if (cfg.curriculum_enabled and passed
                    and state.promotion_streak >= getattr(cfg, "promote_windows", 1)
                    and state.level < state.num_levels - 1):
                state.level += 1
                state.promotion_streak = 0
            state.direction_episodes = [0] * 4
            state.direction_successes = [0] * 4
            state.window_episodes = 0
            state.window_successes = 0
    return {"level": float(state.level), "success_rate": state.last_rate,
            "window_episodes": float(state.window_episodes),
            "promotion_streak": float(state.promotion_streak)}
