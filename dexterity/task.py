import abc
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_env import specs

from dexterity import effector
from dexterity import goal
from dexterity.models.hands import dexterous_hand


class GoalObservable(observable.Observable):
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


class GoalTask(abc.ABC, composer.Task):
    """Base class for goal-based dexterous manipulation tasks."""

    def __init__(
        self,
        arena: composer.Arena,
        hand: dexterous_hand.DexterousHand,
        hand_effector: effector.Effector,
        goal_generator: goal.GoalGenerator,
        success_threshold: float,
        successes_needed: int = 1,
        steps_before_changing_goal: int = 0,
        max_time_per_goal: Optional[float] = None,
    ) -> None:
        self._arena = arena
        self._hand = hand
        self._hand_effector = hand_effector
        self._goal_generator = goal_generator
        self._steps_before_changing_goal = steps_before_changing_goal
        self._successes_needed = successes_needed
        self._max_time_per_goal = max_time_per_goal
        self._success_threshold = success_threshold

        self._goal = np.empty(())  # Dummy value.

    def after_compile(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del physics, random_state  # Unused.

        self._hand_effector.after_compile(self.root_entity.mjcf_model.root)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._hand_effector.initialize_episode(physics, random_state)
        self._goal_generator.initialize_episode(physics, random_state)

        # Generate the first goal.
        self._goal = self._goal_generator.next_goal(physics, random_state)

        self._successes = 0
        self._success_change_counter = 0
        self._solve_start_time = physics.data.time
        self._exceeded_single_goal_time = False
        self._success_registered = False
        self._goal_changed = True

    def before_step(self, physics, action, random_state):
        if self._success_change_counter >= self._steps_before_changing_goal:
            self._goal = self._goal_generator.next_goal(physics, random_state)
            self._success_change_counter = 0
            self._exceeded_single_goal_time = False
            self._solve_start_time = physics.data.time
            self._goal_changed = True
            self._success_registered = False
        else:
            self._goal_changed = False

        self._hand_effector.set_control(physics, action)

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.

        self._goal_distance = self._goal_generator.goal_distance(
            self._goal,
            self._goal_generator.current_state(physics),
        )

        if np.all(self._goal_distance <= self._success_threshold):
            self._success_change_counter += 1
            if not self._success_registered:
                self._successes += 1
                self._success_registered = True
        else:
            if self._max_time_per_goal is not None:
                if physics.data.time - self._solve_start_time > self._max_time_per_goal:
                    self._exceeded_single_goal_time = True

    def should_terminate_episode(self, physics):
        del physics  # Unused.
        if self._successes >= self._successes_needed or (
            self._max_time_per_goal is not None and self._exceeded_single_goal_time
        ):
            return True
        return False

    def get_discount(self, physics: mjcf.Physics) -> float:
        # In the finite-horizon setting, on successful termination, we return 0.0 to
        # indicate a terminal state. If the episode did not successfully terminate,
        # i.e., the agent exceeded the time limit for a single solve, we return a
        # discount of 1.0 to indicate that the agent should treat the episode as if it
        # would have continued, even though the trajectory is truncated.
        del physics  # Unused.
        if self._successes >= self._successes_needed:
            return 0.0
        return 1.0

    @property
    def task_observables(self) -> OrderedDict:
        task_observables = OrderedDict()

        def _get_goal(physics: mjcf.Physics) -> np.ndarray:
            del physics  # Unused.
            return np.array(self._goal)

        goal_spec = self._goal_generator.goal_spec()
        goal_observable = GoalObservable(_get_goal, goal_spec)
        goal_observable.enabled = True
        task_observables["goal_state"] = goal_observable

        return task_observables

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return self._hand_effector.action_spec(physics)

    @property
    def step_limit(self) -> Optional[int]:
        """The maximum number of steps in an episode."""
        return None

    @property
    def time_limit(self) -> float:
        """The maximum number of seconds in an episode."""
        return float("inf")

    @property
    def successes(self) -> int:
        return self._successes

    @property
    def successes_needed(self) -> int:
        return self._successes_needed

    @property
    def goal_generator(self) -> goal.GoalGenerator:
        return self._goal_generator

    @property
    def hand(self) -> dexterous_hand.DexterousHand:
        return self._hand
