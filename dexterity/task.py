from collections import OrderedDict
from typing import Optional

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_env import specs

from dexterity import effector
from dexterity import goal
from dexterity.models.hands import dexterous_hand


class Task(composer.Task):
    """Base class for dexterous manipulation tasks.

    This class overrides the `before_step` method by delegating the actuation to the
    `hand_effector`.
    """

    def __init__(
        self,
        arena: composer.Arena,
        hand: dexterous_hand.DexterousHand,
        hand_effector: effector.Effector,
    ) -> None:
        self._arena = arena
        self._hand = hand
        self._hand_effector = hand_effector

    # Composer overrides.

    def after_compile(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del physics, random_state  # Unused.

        self._hand_effector.after_compile(self.root_entity.mjcf_model.root)

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._hand_effector.initialize_episode(physics, random_state)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        del random_state  # Unused.

        self._hand_effector.set_control(physics, action)

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return self._hand_effector.action_spec(physics)

    # Accessors.

    @property
    def root_entity(self) -> composer.Entity:
        return self._arena

    @property
    def arena(self) -> composer.Arena:
        return self._arena

    @property
    def hand(self) -> dexterous_hand.DexterousHand:
        return self._hand

    @property
    def hand_effector(self) -> effector.Effector:
        return self._hand_effector

    @property
    def step_limit(self) -> Optional[int]:
        """The maximum number of steps in an episode."""
        return None

    @property
    def time_limit(self) -> float:
        """The maximum number of seconds in an episode."""
        return float("inf")


class GoalTask(Task):
    """Goal reaching based tasks."""

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
        super().__init__(arena, hand, hand_effector)

        self._goal_generator = goal_generator
        self._steps_before_changing_goal = steps_before_changing_goal
        self._successes_needed = successes_needed
        self._max_time_per_goal = max_time_per_goal
        self._success_threshold = success_threshold

        # Initialize with dummy goal to appease `task_observables`.
        self._goal = self._goal_generator.goal_spec().generate_value()

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        super().initialize_episode(physics, random_state)

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
        super().before_step(physics, action, random_state)

        if self._success_change_counter >= self._steps_before_changing_goal:
            self._goal = self._goal_generator.next_goal(physics, random_state)
            self._success_change_counter = 0
            self._exceeded_single_goal_time = False
            self._solve_start_time = physics.data.time
            self._goal_changed = True
            self._success_registered = False
        else:
            self._goal_changed = False

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

        # Add the goal at the current timestep to the task observables.
        goal_spec = self._goal_generator.goal_spec()
        goal_observable = goal.GoalObservable(lambda _: np.array(self._goal), goal_spec)
        goal_observable.enabled = True
        task_observables["goal_state"] = goal_observable

        return task_observables

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        return self._hand_effector.action_spec(physics)

    @property
    def goal_generator(self) -> goal.GoalGenerator:
        return self._goal_generator

    @property
    def successes(self) -> int:
        return self._successes

    @property
    def successes_needed(self) -> int:
        return self._successes_needed
