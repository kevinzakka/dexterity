from typing import Optional

import numpy as np
from dm_control import mjcf
from dm_control.composer.initializers import utils
from dm_env import specs

from dexterity import exception
from dexterity import goal
from dexterity.models.hands import dexterous_hand
from dexterity.utils import mujoco_collisions
from dexterity.utils import mujoco_utils

_REJECTION_SAMPLING_FAILED = (
    "Failed to find a collision-free initial configuration for the fingertips after "
    "{max_rejection_samples} randomly sampled joint configurations."
)


class FingertipCartesianPosition(goal.GoalGenerator):
    def __init__(
        self,
        hand: dexterous_hand.DexterousHand,
        ignore_self_collisions: bool = False,
        max_rejection_samples: int = 100,
        scale: float = 0.1,
        name: str = "fingertip_position_goal_generator",
    ) -> None:
        super().__init__()

        self._hand = hand
        self._ignore_self_collisions = ignore_self_collisions
        self._max_rejection_samples = max_rejection_samples
        self._scale = scale
        self._name = name

        self._qpos: Optional[np.ndarray] = None
        self._reference_qpos: Optional[np.ndarray] = None

    def goal_spec(self) -> specs.Array:
        return specs.Array(shape=(15,), dtype=np.float64)

    def _has_self_collisions(self, physics: mjcf.Physics) -> bool:
        """Returns True if the hand is in a self-collision state."""
        return mujoco_collisions.has_self_collision(physics, self._hand.name)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.

        # Apply gravity compensation.
        mujoco_utils.compensate_gravity(physics, self._hand.mjcf_model.find_all("body"))

    def current_state(self, physics: mjcf.Physics) -> np.ndarray:
        return np.array(physics.bind(self._hand.fingertip_sites).xpos).ravel()

    def next_goal(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> np.ndarray:
        joint_binding = physics.bind(self._hand.joints)
        actuator_binding = physics.bind(self._hand.actuators)

        # Get joint limits and range.
        joint_limits = joint_binding.range
        joint_range = joint_limits[:, 1] - joint_limits[:, 0]

        # Use the midrange of the joint angles as the reference configuration.
        if self._reference_qpos is None:
            self._reference_qpos = joint_limits.mean(axis=1)

        initial_qpos = joint_binding.qpos.copy()
        initial_ctrl = actuator_binding.ctrl.copy()
        fingertip_pos: Optional[np.ndarray] = None

        for _ in range(self._max_rejection_samples):
            # Sample around the reference configuration in joint space.
            qpos_desired = random_state.normal(
                loc=self._reference_qpos, scale=self._scale * joint_range
            )
            np.clip(
                qpos_desired, joint_limits[:, 0], joint_limits[:, 1], out=qpos_desired
            )

            self._hand.set_joint_angles(physics, qpos_desired)
            physics.forward()

            # Take a few steps to avoid goals that are impossible due to contact.
            ctrl_desired = self._hand.joint_positions_to_control(qpos_desired)
            actuator_binding.ctrl[:] = ctrl_desired
            with utils.JointStaticIsolator(physics, self._hand.joints):
                for _ in range(2):
                    physics.step()

            if self._ignore_self_collisions or not self._has_self_collisions(physics):
                qpos_desired = joint_binding.qpos.copy()
                fingertip_pos = physics.bind(self._hand.fingertip_sites).xpos.copy()
                self._qpos = qpos_desired
                break

        # Restore the initial joint configuration and ctrl.
        actuator_binding.ctrl[:] = initial_ctrl
        self._hand.set_joint_angles(physics, initial_qpos)
        physics.forward()

        if fingertip_pos is None:
            raise exception.GoalInitializationError(
                _REJECTION_SAMPLING_FAILED.format(
                    max_rejection_samples=self._max_rejection_samples
                )
            )
        else:
            return fingertip_pos.ravel()

    def relative_goal(
        self, goal_state: np.ndarray, current_state: np.ndarray
    ) -> np.ndarray:
        return goal_state - current_state

    def goal_distance(
        self, goal_state: np.ndarray, current_state: np.ndarray
    ) -> np.ndarray:
        relative_goal = self.relative_goal(goal_state, current_state).reshape(-1, 3)
        return np.linalg.norm(relative_goal, axis=1)

    @property
    def name(self) -> str:
        return self._name

    @property
    def qpos(self) -> Optional[np.ndarray]:
        """The joint configuration used to place the target sites."""
        return self._qpos
