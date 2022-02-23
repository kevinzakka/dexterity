from typing import Optional, Union

import enum
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_env import specs

from shadow_hand.models.hands import shadow_hand_e_constants as consts
from shadow_hand.ik import ik_solver


class ActionSpace(enum.Enum):
    """The action space of the hand."""

    JOINT_POSITION = enum.auto()
    """The hand is actuated via delta joint positions."""

    FINGERTIP_POSE = enum.auto()
    """The hand is actuated by specifying fingertip Cartesian poses. The poses are
    converted to joint positions using inverse kinematics under the hood."""


class Task(control.Task):
    """Base class for tasks in the Shadow Hand Control Suite."""

    def __init__(
        self,
        action_space: ActionSpace,
        random: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        """Constructor.

        Args:
            action_space: Instance of `ActionSpace` specifying the action space for the
                hand.
            random: A `numpy.random.RandomState` instance or None. If `None`, a seed is
                automatically selected.
        """
        self._action_space = action_space
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)
        self._random = random

        if action_space == ActionSpace.FINGERTIP_POSE:
            self._setup_ik()

    @property
    def random(self) -> np.random.RandomState:
        return self._random

    def before_step(self, action: np.ndarray, physics: mujoco.Physics) -> None:
        physics.set_control(action)

    def action_spec(self, physics: mujoco.Physics) -> specs.BoundedArray:
        return mujoco.action_spec(physics)

    def _setup_ik(self) -> None:
        """Sets up the inverse kinematics solver."""
        # We want to control all the fingers.
        fingers = list(consts.Components)

        solver = ik_solver.IKSolver(
            model=self._arena.mjcf_model,
            fingers=fingers,
            prefix=self._hand.mjcf_model.model,
        )
