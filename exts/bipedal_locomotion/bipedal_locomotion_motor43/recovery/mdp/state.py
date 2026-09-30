"""Per-environment recovery bookkeeping; no simulator stepping inside resets."""

import torch
from collections import deque


class GetUpState:
    def __init__(self, env):
        self.num_levels = len(getattr(env.cfg.getup, "tilt_ranges_deg", (0, 1, 2, 3)))
        self.level = env.cfg.getup.initial_level
        self.promotion_streak = 0
        self.metric_windows = deque(maxlen=256)
        self.direction_episodes = [0] * 4
        self.direction_successes = [0] * 4
        self.direction = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.window_episodes = 0
        self.window_successes = 0
        self.last_rate = 0.0
        self.stage = torch.full((env.num_envs,), self.level, device=env.device, dtype=torch.long)
        self.hold = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.steps = torch.zeros_like(self.hold)
        self.active = torch.zeros_like(self.success)
        self.landing_steps = torch.zeros_like(self.hold)
        self.control_ready = torch.ones_like(self.success)
        self.start_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self.last_step = -1

    def state_dict(self):
        # Only curriculum statistics persist. Physical episodes restart on resume.
        return {"level": self.level, "window_episodes": self.window_episodes,
                "window_successes": self.window_successes, "last_rate": self.last_rate,
                "promotion_streak": self.promotion_streak,
                "direction_episodes": list(self.direction_episodes),
                "direction_successes": list(self.direction_successes)}

    def load_state_dict(self, data):
        level = int(data["level"])
        if level not in range(self.num_levels):
            raise ValueError(f"Invalid GetUp curriculum level: {level}")
        self.level = level
        self.window_episodes = int(data["window_episodes"])
        self.window_successes = int(data["window_successes"])
        self.last_rate = float(data["last_rate"])
        self.promotion_streak = int(data.get("promotion_streak", 0))
        self.direction_episodes = list(data.get("direction_episodes", [0] * 4))
        self.direction_successes = list(data.get("direction_successes", [0] * 4))


def get_state(env):
    if not hasattr(env, "task_state"):
        env.task_state = GetUpState(env)
    return env.task_state
