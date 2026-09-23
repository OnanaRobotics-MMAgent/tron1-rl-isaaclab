"""Position action with a final clamp to the articulation's physical joint limits."""

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions import JointVelocityAction

from .state import get_state


class LimitedLegPositionAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._processed_actions.clamp_(min=limits[..., 0], max=limits[..., 1])


class LandingLegPositionAction(LimitedLegPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        waiting = ~get_state(self._env).control_ready
        self._processed_actions[waiting] = self._asset.data.default_joint_pos[waiting][:, self._joint_ids]


class LandingWheelVelocityAction(JointVelocityAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions[~get_state(self._env).control_ready] = 0.0
