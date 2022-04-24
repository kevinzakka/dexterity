"""Tasks involving hand finger reaching."""

import dataclasses
from typing import Optional

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.utils import containers

from dexterity import effector
from dexterity import effectors
from dexterity import goal
from dexterity import task
from dexterity.manipulation import arenas
from dexterity.manipulation.goals import fingertip_position
from dexterity.manipulation.props import TargetSphere
from dexterity.manipulation.shared import cameras
from dexterity.manipulation.shared import observations
from dexterity.manipulation.shared import rewards
from dexterity.manipulation.shared import tags
from dexterity.models.hands import adroit_hand
from dexterity.models.hands import adroit_hand_constants as consts
from dexterity.models.hands import dexterous_hand

# The position of the hand relative in the world frame, in meters.
_HAND_POS = (0.0, 0.2, 0.1)
# The orientation of the hand relative to the world frame.
_HAND_QUAT = (0.0, 0.0, 0.707106781186, -0.707106781186)

_SITE_SIZE = 1e-2
_SITE_ALPHA = 0.1
_SITE_COLORS = (
    (1.0, 0.0, 0.0),  # Red.
    (0.0, 1.0, 0.0),  # Green.
    (0.0, 0.0, 1.0),  # Blue.
    (0.0, 1.0, 1.0),  # Cyan.
    (1.0, 0.0, 1.0),  # Magenta.
    (1.0, 1.0, 0.0),  # Yellow.
)
_TARGET_SIZE = 5e-3
_TARGET_ALPHA = 1.0

_STEPS_BEFORE_MOVING_TARGET: int = 5

# Threshold for the distance between a finger and its target below which we consider the
# target reached, in meters.
# Note: OpenAI uses a threshold of 0.025.
_DISTANCE_TO_TARGET_THRESHOLD = 0.01  # 1cm.

# Assign this color to the finger geoms if the finger is within the target threshold.
_THRESHOLD_COLOR = (0.0, 1.0, 0.0)  # Green.

# Timestep of the physics simulation.
# OpenAI uses a timestep of 0.002.
_PHYSICS_TIMESTEP: float = 0.02

# Interval between agent actions, in seconds.
# We send a control signal every (_CONTROL_TIMESTEP / _PHYSICS_TIMESTEP) physics steps.
# OpeAI uses a control timestep that is 10x the physics timestep.
_CONTROL_TIMESTEP: float = 0.02  # 50 Hz.

# The maximum number of consecutive solves until the task is terminated.
_SUCCESSED_NEEDED: int = 50

# The maximum allowed time for reaching the current target, in seconds.
_MAX_STEPS_SINGLE_SOLVE: int = 150
_MAX_TIME_SINGLE_SOLVE: float = _MAX_STEPS_SINGLE_SOLVE * _CONTROL_TIMESTEP

SUITE = containers.TaggedTasks()


class Reach(task.GoalTask):
    """Move the fingers to desired goal positions."""

    def __init__(
        self,
        arena: composer.Arena,
        hand: dexterous_hand.DexterousHand,
        hand_effector: effector.Effector,
        goal_generator: goal.GoalGenerator,
        use_dense_reward: bool,
        visualize_reward: bool,
        success_threshold: float = _DISTANCE_TO_TARGET_THRESHOLD,
        successes_needed: int = _SUCCESSED_NEEDED,
        steps_before_changing_goal: int = _STEPS_BEFORE_MOVING_TARGET,
        max_time_per_goal: Optional[float] = _MAX_TIME_SINGLE_SOLVE,
        control_timestep: float = _CONTROL_TIMESTEP,
        physics_timestep: float = _PHYSICS_TIMESTEP,
    ) -> None:
        """Construct a new `Reach` task."""

        super().__init__(
            arena=arena,
            hand=hand,
            hand_effector=hand_effector,
            goal_generator=goal_generator,
            success_threshold=success_threshold,
            successes_needed=successes_needed,
            steps_before_changing_goal=steps_before_changing_goal,
            max_time_per_goal=max_time_per_goal,
        )

        self._use_dense_reward = use_dense_reward
        self._visualize_reward = visualize_reward

        # Attach the hand to the arena.
        arena.attach_offset(hand, position=_HAND_POS, quaternion=_HAND_QUAT)

        # Make the hand fingertip sites visible and recolor them.
        for i, site in enumerate(hand.fingertip_sites):
            site.group = None  # Make the sites visible.
            site.size = (_SITE_SIZE,) * 3  # Increase their size.
            site.rgba = _SITE_COLORS[i] + (_SITE_ALPHA,)  # Change their color.

        # Create fingertip targets and attach them to the arena.
        self._targets = []
        for i, site in enumerate(hand.fingertip_sites):
            target = TargetSphere(
                radius=_TARGET_SIZE,
                rgba=_SITE_COLORS[i] + (_TARGET_ALPHA,),
                name=f"target_{site.name}",
            )
            arena.attach(target)
            self._targets.append(target)

        # Disable collisions for the ground plane. It's only here for visualization
        # purposes.
        arena.ground.contype = 0
        arena.ground.conaffinity = 0

        # Add a closeup camera, used for rendering.
        arena.mjcf_model.worldbody.add(
            "camera", **dataclasses.asdict(cameras.FRONT_CLOSE)
        )

        self.set_timesteps(control_timestep, physics_timestep)

    @property
    def root_entity(self) -> composer.Entity:
        return self._arena

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        super().initialize_episode(physics, random_state)

        # Set the initial joint configuration to the midrange of the joint limits.
        midrange = physics.bind(self.hand.joints).range.mean(axis=1)
        physics.bind(self.hand.joints).qpos[:] = midrange

        # Step the physics to move the fingers out of the way. Typically the pinky
        # collides with the ring finger in this configuration.
        for _ in range(2):
            physics.step()

        for i, target in enumerate(self._targets):
            physics.bind(target.site).pos = self._goal[i]

        # Save initial finger colors.
        if self._visualize_reward:
            self._init_finger_colors = {}
            for i, geoms in enumerate(consts.FINGER_GEOM_MAPPING.values()):
                elems = [
                    elem
                    for elem in self._hand.mjcf_model.find_all("geom")
                    if elem.name in geoms
                ]
                self._init_finger_colors[i] = (elems, physics.bind(elems).rgba)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        super().before_step(physics, action, random_state)

        if self._goal_changed:
            for i, target in enumerate(self._targets):
                physics.bind(target.site).pos = self._goal[i]

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        super().after_step(physics, random_state)

        if self._visualize_reward:
            self._maybe_color_fingers(physics)

    def get_reward(self, physics: mjcf.Physics) -> float:
        del physics  # Unused.
        if self._use_dense_reward:
            # Dense reward.
            return np.mean(
                np.where(
                    self._goal_distance <= _DISTANCE_TO_TARGET_THRESHOLD,
                    0.0,
                    [-rewards.tanh_squared(d, margin=0.1) for d in self._goal_distance],
                )
            )
        # Sparse reward.
        return np.mean(
            np.where(self._goal_distance <= _DISTANCE_TO_TARGET_THRESHOLD, 0.0, -1.0)
        )

    # Helper methods.

    def _maybe_color_fingers(self, physics: mjcf.Physics) -> None:
        for i, distance in enumerate(self._goal_distance):
            elems, rgba = self._init_finger_colors[i]
            if distance <= self._success_threshold:
                physics.bind(elems).rgba = _THRESHOLD_COLOR + (1.0,)
            else:
                physics.bind(elems).rgba = rgba


def reach_task(
    observation_set: observations.ObservationSet,
    use_dense_reward: bool,
    visualize_reward: bool = True,
) -> task.GoalTask:
    """Configure and instantiate a `Reach` task."""
    arena = arenas.Standard()

    hand = adroit_hand.AdroitHand(
        observable_options=observations.make_options(
            observation_set.value,
            observations.HAND_OBSERVABLES,
        ),
    )

    hand_effector = effectors.HandEffector(hand=hand, hand_name=hand.name)

    goal_generator = fingertip_position.FingertipCartesianPosition(
        hand=hand,
        ignore_self_collisions=False,
    )

    return Reach(
        arena=arena,
        hand=hand,
        hand_effector=hand_effector,
        goal_generator=goal_generator,
        use_dense_reward=use_dense_reward,
        visualize_reward=visualize_reward,
    )


@SUITE.add(tags.STATE, tags.DENSE)
def state_dense() -> task.GoalTask:
    """Reach task with full state observations and dense reward."""
    return reach_task(
        observation_set=observations.ObservationSet.STATE_ONLY,
        use_dense_reward=True,
        visualize_reward=True,
    )


@SUITE.add(tags.STATE, tags.SPARSE)
def state_sparse() -> task.GoalTask:
    """Reach task with full state observations and sparse reward."""
    return reach_task(
        observation_set=observations.ObservationSet.STATE_ONLY,
        use_dense_reward=False,
        visualize_reward=True,
    )
