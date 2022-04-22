"""Shared initializers for the hand."""

from typing import Optional, Sequence

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.initializers import utils

from dexterity import hints
from dexterity.models.hands import dexterous_hand
from dexterity.utils import mujoco_collisions
from dexterity.utils import mujoco_utils

_REJECTION_SAMPLING_FAILED = (
    "Failed to find a collision-free initial configuration for the fingertips after "
    "{max_rejection_samples} randomly sampled joint configurations."
)


class FingertipPositionPlacer(composer.Initializer):
    """An initializer that sets target site positions for the fingertips of a hand.

    This initializer works backwards by sampling a joint configuration for the entire
    hand, then querying the fingertip positions using forward kinematics. This ensures
    that the fingertip sites are always reachable.
    """

    def __init__(
        self,
        target_sites: Sequence[hints.MjcfElement],
        hand: dexterous_hand.DexterousHand,
        ignore_self_collisions: bool = False,
        max_rejection_samples: int = 100,
        scale: float = 0.1,
    ) -> None:
        """Constructor.

        Args:
            target_sites: The target fingertip sites to place.
            hand: An instance of `fingered_hand.FingeredHand`.
            ignore_self_collisions: If True, self-collisions are ignored, i.e.,
                rejection sampling is disabled.
            max_rejection_samples: The maximum number of joint configurations to sample
                while attempting to find a collision-free configuration.
            scale: Standard deviation of the Gaussian noise added to the current joint
                configuration to sample the goal joint configuration, expressed as a
                fraction of the max-min range for each joint angle.
        """
        super().__init__()

        self._target_sites = target_sites
        self._hand = hand
        self._ignore_self_collisions = ignore_self_collisions
        self._max_rejection_samples = max_rejection_samples
        self._scale = scale
        self._qpos = None

    def _has_self_collisions(self, physics: mjcf.Physics) -> bool:
        """Returns True if the hand is in a self-collision state."""
        return mujoco_collisions.has_self_collision(physics, self._hand.name)

    def __call__(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Sets the position of the fingertip target sites.

        Raises:
            RuntimeError: If a collision-free configuration of the fingertips could not
                be found after `max_rejection_samples` randomly sampled joint
                configurations.
        """
        joint_binding = physics.bind(self._hand.joints)
        actuator_binding = physics.bind(self._hand.actuators)

        initial_qpos = joint_binding.qpos.copy()
        initial_ctrl = actuator_binding.ctrl.copy()
        fingertip_pos = None

        # Get joint limits and range.
        joint_limits = joint_binding.range
        joint_range = joint_limits[:, 1] - joint_limits[:, 0]

        # Apply gravity compensation.
        mujoco_utils.compensate_gravity(physics, self._hand.mjcf_model.find_all("body"))

        for _ in range(self._max_rejection_samples):
            # Sample around the current configuration in joint space.
            qpos_desired = random_state.normal(
                loc=initial_qpos, scale=self._scale * joint_range
            )
            np.clip(
                qpos_desired, joint_limits[:, 0], joint_limits[:, 1], out=qpos_desired
            )

            self._hand.set_joint_angles(physics, qpos_desired)
            physics.forward()

            if self._ignore_self_collisions or not self._has_self_collisions(physics):
                ctrl_desired = self._hand.joint_positions_to_control(qpos_desired)
                actuator_binding.ctrl[:] = ctrl_desired

                # Take a few steps to avoid goals that are impossible due to contact.
                with utils.JointStaticIsolator(physics, self._hand.joints):
                    for _ in range(2):
                        physics.step()

                qpos_desired = joint_binding.qpos.copy()
                fingertip_pos = physics.bind(self._hand.fingertip_sites).xpos.copy()
                self._qpos = qpos_desired
                break

        # Restore the initial joint configuration and ctrl.
        actuator_binding.ctrl[:] = initial_ctrl
        self._hand.set_joint_angles(physics, initial_qpos)
        physics.forward()

        if fingertip_pos is None:
            raise RuntimeError(
                _REJECTION_SAMPLING_FAILED.format(
                    max_rejection_samples=self._max_rejection_samples
                )
            )
        else:
            physics.bind(self._target_sites).pos = fingertip_pos

    @property
    def qpos(self) -> Optional[np.ndarray]:
        """The joint configuration used to place the target sites."""
        return self._qpos
