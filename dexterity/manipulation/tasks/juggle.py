"""Tasks involving juggling."""

import dataclasses
from typing import Sequence

import numpy as np
from dm_control import mjcf
from dm_control.utils import containers

from dexterity import effector
from dexterity import effectors
from dexterity import task
from dexterity.manipulation import props
from dexterity.manipulation.shared import cameras
from dexterity.manipulation.shared import observations
from dexterity.manipulation.shared import tags
from dexterity.models import arenas
from dexterity.models.hands import dexterous_hand
from dexterity.models.hands import mpl_hand

# The orientation of the hand relative to the world frame.
_HAND_QUAT = (0.0, 0.0, 0.7, 0.0)

# The position of the hand relative in the world frame, in meters.
_RIGHT_HAND_POS = (-0.1, 0.0, 0.1)
_LEFT_HAND_POS = (0.1, 0.0, 0.1)

_LEFT_MOCAP_COLOR = (0.9, 0.5, 0.5, 1.0)
_RIGHT_MOCAP_COLOR = (0.5, 0.9, 0.5, 1.0)

_BALL_RADIUS = 0.025

# Timestep of the physics simulation.
# OpenAI uses a timestep of 0.002.
_PHYSICS_TIMESTEP: float = 0.02

# Interval between agent actions, in seconds.
# We send a control signal every (_CONTROL_TIMESTEP / _PHYSICS_TIMESTEP) physics steps.
# OpeAI uses a control timestep that is 10x the physics timestep.
_CONTROL_TIMESTEP: float = 0.02  # 50 Hz.

SUITE = containers.TaggedTasks()


class Juggle(task.Task):
    """Juggle a ball with two hands."""

    def __init__(
        self,
        arena: arenas.Arena,
        hands: Sequence[dexterous_hand.DexterousHand],
        hand_effectors: Sequence[effector.Effector],
        use_dense_reward: bool,
        control_timestep: float = _CONTROL_TIMESTEP,
        physics_timestep: float = _PHYSICS_TIMESTEP,
    ) -> None:
        """Construct a new `Juggle` task."""

        hand_names = set([hand.name for hand in hands])
        if len(hand_names) != len(hands):
            raise ValueError("Hands must have unique names.")

        super().__init__(arena=arena, hands=hands, hand_effectors=hand_effectors)

        self._use_dense_reward = use_dense_reward

        # Attach the hand to the arena.
        self._left_mocap = arena.add_mocap(
            hands[0],
            position=_LEFT_HAND_POS,
            quaternion=_HAND_QUAT,
            name="left_mocap",
            visible=True,
            color=_LEFT_MOCAP_COLOR,
        )
        self._right_mocap = arena.add_mocap(
            hands[1],
            position=_RIGHT_HAND_POS,
            quaternion=_HAND_QUAT,
            name="right_mocap",
            visible=True,
            color=_RIGHT_MOCAP_COLOR,
        )

        # Add a ball.
        ball = props.JugglingBall(radius=_BALL_RADIUS)
        arena.add_free_entity(ball)
        self._ball = ball

        # Add a closeup camera, used for rendering.
        arena.mjcf_model.worldbody.add(
            "camera", **dataclasses.asdict(cameras.FRONT_CLOSE)
        )

        self.set_timesteps(control_timestep, physics_timestep)

    @property
    def left_hand(self) -> dexterous_hand.DexterousHand:
        return self.hands[0]

    @property
    def right_hand(self) -> dexterous_hand.DexterousHand:
        return self.hands[1]

    @property
    def ball(self):
        return self._ball

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        super().initialize_episode(physics, random_state)

        # Set the initial joint configuration to the midrange of the joint limits.
        midrange = physics.bind(self.left_hand.joints).range.mean(axis=1)
        physics.bind(self.left_hand.joints).qpos[:] = midrange

        midrange = physics.bind(self.right_hand.joints).range.mean(axis=1)
        physics.bind(self.right_hand.joints).qpos[:] = midrange

        # Step the physics to move the fingers out of the way. Typically the pinky
        # collides with the ring finger in this configuration.
        for _ in range(2):
            physics.step()

        # Place the ball on the palm of the left hand.
        palm = self.left_hand.mjcf_model.find("body", "palm")
        palm_position = physics.bind(palm).xpos
        ball_position = palm_position.copy()
        ball_position[1] -= 0.05
        ball_position[2] += 0.05
        self.ball.set_pose(physics, position=ball_position)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        super().before_step(physics, action, random_state)

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        super().after_step(physics, random_state)

    def get_reward(self, physics: mjcf.Physics) -> float:
        return 0.0


def juggle_task(
    observation_set: observations.ObservationSet,
    use_dense_reward: bool,
) -> task.Task:
    """Configure and instantiate a `Juggle` task."""
    arena = arenas.Standard()

    left_hand = mpl_hand.MPLHand(
        side=dexterous_hand.HandSide.LEFT,
        observable_options=observations.make_options(
            observation_set.value,
            observations.HAND_OBSERVABLES,
        ),
    )
    left_hand_effector = effectors.HandEffector(left_hand, left_hand.name)

    right_hand = mpl_hand.MPLHand(
        side=dexterous_hand.HandSide.RIGHT,
        observable_options=observations.make_options(
            observation_set.value,
            observations.HAND_OBSERVABLES,
        ),
    )
    right_hand_effector = effectors.HandEffector(right_hand, right_hand.name)

    return Juggle(
        arena=arena,
        hands=[left_hand, right_hand],
        hand_effectors=[left_hand_effector, right_hand_effector],
        use_dense_reward=use_dense_reward,
    )


@SUITE.add(tags.STATE, tags.SPARSE)
def state_sparse() -> task.GoalTask:
    """Juggle task with full state observations and sparse reward."""
    return juggle_task(
        observation_set=observations.ObservationSet.STATE_ONLY,
        use_dense_reward=False,
    )
