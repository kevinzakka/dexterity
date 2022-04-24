import abc
from typing import Callable

import numpy as np
from dm_control import mjcf
from dm_control.composer.observation.observable import Observable
from dm_env import specs


class GoalObservable(Observable):
    def __init__(
        self,
        raw_observation_callable: Callable[[mjcf.Physics], np.ndarray],
        goal_spec: specs.Array,
        update_interval=1,
        buffer_size=None,
        delay=None,
        aggregator=None,
        corruptor=None,
    ) -> None:

        self._raw_callable = raw_observation_callable
        self._goal_spec = goal_spec

        super().__init__(update_interval, buffer_size, delay, aggregator, corruptor)

    @property
    def array_spec(self) -> specs.Array:
        return self._goal_spec

    def _callable(self, physics: mjcf.Physics) -> Callable[[], np.ndarray]:
        return lambda: self._raw_callable(physics)


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
