"""Tasks involving in-hand object re-orientation."""

import dataclasses
import random
from typing import Dict, List

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.transformations import transformations as tr

from shadow_hand import effector
from shadow_hand import effectors
from shadow_hand import hints
from shadow_hand import task
from shadow_hand.models.hands import fingered_hand
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.tasks.inhand_manipulation.shared import arenas
from shadow_hand.tasks.inhand_manipulation.shared import constants
from shadow_hand.tasks.inhand_manipulation.shared import initializers
from shadow_hand.tasks.inhand_manipulation.shared import observations
from shadow_hand.tasks.inhand_manipulation.shared import registry
from shadow_hand.tasks.inhand_manipulation.shared import tags
from shadow_hand.tasks.inhand_manipulation.shared import workspaces

# The position of the hand relative in the world frame, in meters.
_HAND_POS = (0, 0.2, 0.1)
# The orientation of the hand relative to the world frame.
_HAND_QUAT = tr.axisangle_to_quat(
    np.pi * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
)

_SITE_COLORS = (
    (1.0, 0.0, 0.0),  # Red.
    (0.0, 1.0, 0.0),  # Green.
    (0.0, 0.0, 1.0),  # Blue.
    (0.0, 1.0, 1.0),  # Cyan.
    (1.0, 0.0, 1.0),  # Magenta.
    (1.0, 1.0, 0.0),  # Yellow.
)
_SITE_SIZE = 1e-2
_SITE_ALPHA = 0.1
_TARGET_SIZE = 5e-3
_TARGET_ALPHA = 1.0

# Observable settings.
_HAND_OBSERVABLES = observations.ObservableNames(
    proprio=(
        "joint_positions",
        "joint_velocities",
        "fingerip_positions",
        "fingertip_linear_velocities",
    ),
)


class Reach(task.Task):
    """Move the fingers to desired goal positions."""

    def __init__(
        self,
        arena: arenas.Standard,
        hand: fingered_hand.FingeredHand,
        hand_effector: effector.Effector,
        obs_settings: observations.ObservationSettings,
        control_timestep: float = constants.CONTROL_TIMESTEP,
        physics_timestep: float = constants.PHYSICS_TIMESTEP,
    ) -> None:
        """Construct a new `Reach` task.

        Args:
            arena: The arena to use.
            hand: The hand to use.
            hand_effector: The effector to use for the hand.
            obs_settings: The observation settings to use.
            workspace: The workspace to use.
            control_timestep: The control timestep, in seconds.
            physics_timestep: The physics timestep, in seconds.
        """
        super().__init__(arena=arena, hand=hand, hand_effector=hand_effector)

        # Attach the hand to the arena.
        self._arena.attach_offset(hand, position=_HAND_POS, quaternion=_HAND_QUAT)

        self.set_timesteps(
            control_timestep=control_timestep,
            physics_timestep=physics_timestep,
        )

        # Create visible sites for the finger tips.
        color_sequence = random.sample(_SITE_COLORS, k=len(_SITE_COLORS))
        for i, site in enumerate(self._hand.fingertip_sites):
            site.group = None  # Make the sites visible.
            site.size = (_SITE_SIZE,) * 3  # Increase their size.
            site.rgba = color_sequence[i] + (_SITE_ALPHA,)  # Change their color.

        # Create target sites for each fingertip.
        self._target_sites: List[hints.MjcfElement] = []
        for i, site in enumerate(self._hand.fingertip_sites):
            self._target_sites.append(
                workspaces.add_target_site(
                    body=arena.mjcf_model.worldbody,
                    radius=_TARGET_SIZE,
                    visible=True,
                    rgba=color_sequence[i] + (_TARGET_ALPHA,),
                    name=f"target_{site.name}",
                )
            )

        self._fingertips_initializer = initializers.FingertipPositionPlacer(
            target_sites=self._target_sites, hand=hand
        )

        # Add task observables.
        self._task_observables: Dict[str, observable.Observable] = {}

        target_positions_observable = observable.Generic(self._get_target_positions)
        target_positions_observable.configure(
            **dataclasses.asdict(obs_settings.prop_pose)
        )
        self._task_observables["target_positions"] = target_positions_observable

        fingertip_positions_observable = observable.Generic(
            self._get_fingertip_positions
        )
        fingertip_positions_observable.configure(
            **dataclasses.asdict(obs_settings.prop_pose)
        )
        self._task_observables["fingertip_positions"] = fingertip_positions_observable

    @property
    def task_observables(self) -> Dict[str, observable.Observable]:
        return self._task_observables

    @property
    def root_entity(self) -> composer.Entity:
        return self._arena

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._fingertips_initializer(physics, random_state)

    def get_reward(self, physics: mjcf.Physics) -> float:
        fingertip_positions = self._get_fingertip_positions(physics)
        target_positions = self._get_target_positions(physics)
        distance = float(np.linalg.norm(fingertip_positions - target_positions))
        reward = -1.0 * distance
        return reward

    def get_discount(self, physics: mjcf.Physics) -> float:
        del physics  # Unused.
        return 1.0

    # Helper methods.

    def _get_target_positions(self, physics: mjcf.Physics) -> np.ndarray:
        return np.array(physics.bind(self._target_sites).xpos).ravel()

    def _get_fingertip_positions(self, physics: mjcf.Physics) -> np.ndarray:
        return np.array(physics.bind(self._hand.fingertip_sites).xpos).ravel()


def _reach(obs_settings: observations.ObservationSettings) -> composer.Task:
    """Configure and instantiate a `ReOrient` task."""
    arena = arenas.Standard()

    hand = shadow_hand_e.ShadowHandSeriesE(
        observable_options=observations.make_options(obs_settings, _HAND_OBSERVABLES),
    )

    # Effector used for the shadow hand.
    joint_position_effector = effectors.HandEffector(hand=hand, hand_name=hand.name)
    hand_effector = effectors.RelativeToJointPositions(
        joint_position_effector, hand=hand
    )

    return Reach(
        arena=arena,
        hand=hand,
        hand_effector=hand_effector,
        obs_settings=obs_settings,
        control_timestep=constants.CONTROL_TIMESTEP,
        physics_timestep=constants.PHYSICS_TIMESTEP,
    )


@registry.add(tags.FEATURES)
def reach() -> composer.Task:
    return _reach(obs_settings=observations.PERFECT_FEATURES)
