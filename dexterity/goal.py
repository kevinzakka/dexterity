import abc

import numpy as np
from dm_control import mjcf
from dm_env import specs


class GoalGenerator(abc.ABC):
    """Abstract base class for a goal generator."""

    @abc.abstractmethod
    def goal_spec(self) -> specs.Array:
        ...

    @abc.abstractmethod
    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        ...

    @abc.abstractmethod
    def next_goal(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> np.ndarray:
        ...

    @abc.abstractmethod
    def relative_goal(
        self, goal_state: np.ndarray, current_state: np.ndarray
    ) -> np.ndarray:
        ...

    @abc.abstractmethod
    def goal_distance(
        self, goal_state: np.ndarray, current_state: np.ndarray
    ) -> np.ndarray:
        ...

    @abc.abstractmethod
    def current_state(self, physics: mjcf.Physics) -> np.ndarray:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...
