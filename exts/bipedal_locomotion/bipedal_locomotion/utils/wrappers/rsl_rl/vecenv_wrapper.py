"""Adapt both Isaac Lab RSL-RL observation APIs to the vendored encoder runner."""

import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as IsaacLabRslRlVecEnvWrapper


class RslRlVecEnvWrapper(IsaacLabRslRlVecEnvWrapper):
    def get_action_mean_bounds(self):
        """Executable raw-action ranges for GetUp's position/velocity actions."""
        raw = self.unwrapped
        if not hasattr(raw.cfg, "getup"):
            raise ValueError("Automatic action mean bounds require a GetUp task")
        bounds = []
        for name in raw.action_manager.active_terms:
            term = raw.action_manager.get_term(name)
            if term.cfg.clip is None:
                clip = torch.empty(raw.num_envs, term.action_dim, 2, device=raw.device)
                clip[..., 0] = -torch.inf
                clip[..., 1] = torch.inf
            else:
                clip = term._clip.clone()
            if name == "joint_pos":
                physical = term._asset.data.joint_pos_limits[:, term._joint_ids]
                clip[..., 0] = torch.maximum(clip[..., 0], physical[..., 0])
                clip[..., 1] = torch.minimum(clip[..., 1], physical[..., 1])
            scale = torch.as_tensor(term._scale, device=raw.device)
            offset = torch.as_tensor(term._offset, device=raw.device)
            if not bool((scale > 0).all()):
                raise ValueError("GetUp bounds require positive action scales")
            low = (clip[..., 0] - offset) / scale
            high = (clip[..., 1] - offset) / scale
            bounds.append(torch.stack((low.amax(0), high.amin(0)), dim=-1))
        return torch.cat(bounds).cpu().tolist()

    @staticmethod
    def _observations(obs, extras):
        if isinstance(obs, torch.Tensor):
            return obs, extras
        extras = dict(extras)
        extras["observations"] = obs
        return obs["policy"], extras

    def get_observations(self):
        result = super().get_observations()
        if isinstance(result, tuple):
            return result
        return self._observations(result, {})

    def reset(self):
        obs, extras = super().reset()
        return self._observations(obs, extras)

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        obs, extras = self._observations(obs, extras)
        return obs, rewards, dones, extras
