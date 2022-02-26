"""Tasks involviing in-hand object re-orientation."""

import numpy as np
from dm_control import composer
from dm_control.manipulation.shared import observations
from dm_robotics.transformations import transformations as tr

from shadow_hand import hand
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.tasks.inhand_manipulation import (
    arenas,
    cameras,
    constants,
    registry,
    tags,
)


# Alpha value of the visual goal hint representing the goal state for each task.
_HINT_ALPHA = 0.75


class _Common(composer.Task):
    """Manipulate an object until it is in a desired goal configuration."""

    def __init__(
        self,
        arena: composer.Arena,
        hand: hand.Hand,
        control_timestep: float,
    ) -> None:

        self._arena = arena
        self._hand = hand

        axis_angle = np.radians(180) * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
        quat = tr.axisangle_to_quat(axis_angle)
        attachment_site = arena.mjcf_model.worldbody.add(
            "site",
            type="sphere",
            pos=(0, 0.2, 0.1),
            quat=quat,
            rgba="0 0 0 0",
            size="0.01",
        )
        self._arena.attach(hand, attachment_site)

        self._task_observables = cameras.add_camera_observables(
            arena,
            observations.PERFECT_FEATURES,
            cameras.FRONT_CLOSE,
            cameras.TOP_DOWN,
            cameras.LEFT_CLOSE,
            cameras.RIGHT_CLOSE,
        )

        self.control_timestep = control_timestep

    @property
    def root_entity(self) -> composer.Entity:
        return self._arena

    @property
    def hand(self) -> composer.Entity:
        return self._hand

    def get_reward(self, physics) -> float:
        del physics
        return 0.0


class ReOrientSO3(_Common):
    """Manipulate an object to a desired goal configuration sampled from SO(3)."""

    def __init__(self, arena, hand, control_timestep) -> None:
        super().__init__(arena, hand, control_timestep)


class ReOrientZ(_Common):
    def __init__(self, arena, hand, control_timestep) -> None:
        super().__init__(arena, hand, control_timestep)


def _build_arena(name: str) -> composer.Arena:
    arena = arenas.Standard(name)
    arena.mjcf_model.option.timestep = 0.001
    arena.mjcf_model.option.gravity = (0.0, 0.0, -9.81)
    arena.mjcf_model.size.nconmax = 1_000
    arena.mjcf_model.size.njmax = 2_000
    arena.mjcf_model.visual.__getattr__("global").offheight = 480
    arena.mjcf_model.visual.__getattr__("global").offwidth = 640
    return arena


def _reorient_SO3() -> _Common:
    """Configure and instantiate a `_Common` task."""
    arena = _build_arena("arena")
    hand = shadow_hand_e.ShadowHandSeriesE()
    return ReOrientSO3(
        arena=arena,
        hand=hand,
        control_timestep=constants.CONTROL_TIMESTEP,
    )


def _reorient_Z() -> _Common:
    """Configure and instantiate a `_Common` task."""
    arena = _build_arena("arena")
    hand = shadow_hand_e.ShadowHandSeriesE()
    return ReOrientZ(
        arena=arena,
        hand=hand,
        control_timestep=constants.CONTROL_TIMESTEP,
    )


@registry.add(tags.VISION)
def reorient_so3():
    return _reorient_SO3()


@registry.add(tags.VISION)
def reorient_z():
    return _reorient_Z()
