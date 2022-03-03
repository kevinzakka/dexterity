"""Shared initializers for the hand."""

from typing import Sequence

import numpy as np
from dm_control import composer
from dm_control import mjcf

from shadow_hand import hints
from shadow_hand.models.hands import fingered_hand

_REJECTION_SAMPLING_FAILED = (
    "Failed to find a valid initial configuration for the fingertips after "
    "{max_rejection_samples} randomly sampled joint configurations."
)


class FingertipPositionPlacer(composer.Initializer):
    """An initializer that sets the position of the fingertips of the hand."""

    def __init__(
        self,
        target_sites: Sequence[hints.MjcfElement],
        hand: fingered_hand.FingeredHand,
        ignore_collisions: bool = False,
        max_rejection_samples: int = 30,
    ) -> None:
        super().__init__()

        self._target_sites = target_sites
        self._hand = hand
        self._ignore_collisions = ignore_collisions
        self._max_rejection_samples = max_rejection_samples

    def _has_relevant_collisions(self, physics: mjcf.Physics) -> bool:
        mjcf_root = self._hand.mjcf_model.root_model
        all_geoms = mjcf_root.find_all("geom")
        free_body_geoms = set()
        for body in mjcf_root.worldbody.get_children("body"):
            if mjcf.get_freejoint(body):
                free_body_geoms.update(body.find_all("geom"))

        hand_model = self._hand.mjcf_model

        for contact in physics.data.contact:
            geom_1 = all_geoms[contact.geom1]
            geom_2 = all_geoms[contact.geom2]

            # Ignore contacts with positive distance (i.e., not actually touching).
            if contact.dist > 0:
                continue

            if geom_1.root is hand_model and geom_2.root is hand_model:
                return True

        return False

    def _get_fingertip_positions(self, physics: mjcf.Physics) -> np.ndarray:
        return physics.bind(self._hand.fingertip_sites).xpos.copy()

    def __call__(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:

        initial_qpos = physics.bind(self._hand.joints).qpos.copy()
        fingertip_pos = None

        for _ in range(self._max_rejection_samples):
            qpos = random_state.uniform(
                physics.bind(self._hand.joints).range[:, 0],
                physics.bind(self._hand.joints).range[:, 1],
            )
            physics.bind(self._hand.joints).qpos = qpos

            physics.forward()
            if self._ignore_collisions or not self._has_relevant_collisions(physics):
                fingertip_pos = self._get_fingertip_positions(physics)
                break

        physics.bind(self._hand.joints).qpos = initial_qpos
        if fingertip_pos is None:
            raise RuntimeError(
                _REJECTION_SAMPLING_FAILED.format(
                    max_rejection_samples=self._max_rejection_samples
                )
            )
        else:
            physics.bind(self._target_sites).pos = fingertip_pos
