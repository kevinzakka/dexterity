import abc
import dataclasses
import enum
from typing import List, Tuple

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.transformations import transformations as tr

from dexterity.hints import MjcfElement
from dexterity.utils import mujoco_utils


class HandSide(enum.Enum):
    """Which hand side is being modeled."""

    LEFT = enum.auto()
    RIGHT = enum.auto()


@dataclasses.dataclass(frozen=True)
class JointGrouping:
    """A collection of joints belonging to a hand part (wrist or finger)."""

    name: str
    joints: Tuple[MjcfElement, ...]

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return tuple([joint.name for joint in self.joints])


class DexterousHand(abc.ABC, composer.Entity):
    """Base composer class for a dexterous multi-fingered hand."""

    def _build_observables(self) -> composer.Observables:
        return DexterousHandObservables(self)

    @property
    def num_joints(self) -> int:
        return len(self._joints)

    @property
    def num_actuators(self) -> int:
        return len(self._actuators)

    @property
    def underactuated(self) -> bool:
        return self.num_joints > self._num_actuators

    # Abstract properties.

    @property
    @abc.abstractmethod
    def mjcf_model(self) -> mjcf.RootElement:
        """Returns the `mjcf.RootElement` object corresponding to the hand."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the hand."""

    @property
    @abc.abstractmethod
    def joint_groups(self) -> List[JointGrouping]:
        """A list of `JointGrouping` objects corresponding to hand parts."""

    @property
    @abc.abstractmethod
    def joints(self) -> List[MjcfElement]:
        """List of joint elements belonging to the hand."""

    @property
    @abc.abstractmethod
    def actuators(self) -> List[MjcfElement]:
        """List of actuator elements belonging to the hand."""

    @property
    @abc.abstractmethod
    def fingertip_sites(self) -> List[MjcfElement]:
        """List of fingertip site elements belonging to the hand."""

    @property
    @abc.abstractmethod
    def joint_torque_sensors(self) -> List[MjcfElement]:
        """List of joint torque sensor elements belonging to the hand."""

    # Abstract methods.

    @abc.abstractmethod
    def _build(self) -> None:
        """Entity initialization method to be overridden by subclasses."""

    @abc.abstractmethod
    def control_to_joint_positions(self, control: np.ndarray) -> np.ndarray:
        """Maps a control command to a joint position command.

        This method is necessary for underactuated hands.
        """

    @abc.abstractmethod
    def joint_positions_to_control(self, qpos: np.ndarray) -> np.ndarray:
        """Maps a joint position command to a control command.

        This method is necessary for underactuated hands.
        """

    # Note: This method is abstract because a hand might have to perform extra logic
    # specific to its actuation after setting the joint angles.
    @abc.abstractmethod
    def set_joint_angles(self, physics: mjcf.Physics, joint_angles: np.ndarray) -> None:
        """Sets the joints of the hand to a given configuration."""

    # Note: This method is abstract because a hand might have to perform extra logic
    # pertaining to its underactuation after randomly sampling joint angles.
    @abc.abstractmethod
    def sample_joint_angles(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> np.ndarray:
        """Samples a random joint configuration for the hand."""


class DexterousHandObservables(composer.Observables):
    """Observables for a fingered hand."""

    _entity: DexterousHand

    @composer.observable
    def joint_positions(self) -> observable.MJCFFeature:
        return observable.MJCFFeature(kind="qpos", mjcf_element=self._entity.joints)

    @composer.observable
    def joint_positions_sin_cos(self) -> observable.MJCFFeature:
        def _get_joint_pos_sin_cos(physics: mjcf.Physics) -> np.ndarray:
            qpos = physics.bind(self._entity.joints).qpos
            qpos_sin = np.sin(qpos)
            qpos_cos = np.cos(qpos)
            return np.concatenate([qpos_sin, qpos_cos])

        return observable.Generic(raw_observation_callable=_get_joint_pos_sin_cos)

    @composer.observable
    def joint_velocities(self) -> observable.MJCFFeature:
        return observable.MJCFFeature(kind="qvel", mjcf_element=self._entity.joints)

    @composer.observable
    def joint_torques(self) -> observable.Generic:
        def _get_joint_torques(physics: mjcf.Physics) -> np.ndarray:
            torques = physics.bind(self._entity.joint_torque_sensors).sensordata
            joint_axes = physics.bind(self._entity.joints).axis
            joint_torques = np.einsum("ij,ij->i", torques.reshape(-1, 3), joint_axes)
            return joint_torques

        return observable.Generic(raw_observation_callable=_get_joint_torques)

    @composer.observable
    def fingertip_positions(self) -> observable.Generic:
        def _get_fingertip_positions(physics: mjcf.Physics) -> np.ndarray:
            return physics.bind(self._entity.fingertip_sites).xpos.ravel()

        return observable.Generic(raw_observation_callable=_get_fingertip_positions)

    @composer.observable
    def fingertip_orientations(self) -> observable.Generic:
        def _get_fingertip_orientations(physics: mjcf.Physics) -> np.ndarray:
            xmats = physics.bind(self._entity.fingertip_sites).xmat.reshape(-1, 3, 3)
            quats = [tr.mat_to_quat(xmat) for xmat in xmats]
            return np.concatenate(quats)

        return observable.Generic(raw_observation_callable=_get_fingertip_orientations)

    @composer.observable
    def fingertip_linear_velocities(self) -> observable.Generic:
        def _get_fingertip_linear_velocities(physics: mjcf.Physics) -> np.ndarray:
            linear_velocities = []
            for fingertip_site in self._entity.fingertip_sites:
                lin_vel = mujoco_utils.get_site_velocity(physics, fingertip_site)
                linear_velocities.append(lin_vel[:3])
            return np.concatenate(linear_velocities)

        return observable.Generic(
            raw_observation_callable=_get_fingertip_linear_velocities
        )

    @composer.observable
    def fingertip_angular_velocities(self) -> observable.Generic:
        def _get_fingertip_angular_velocities(physics: mjcf.Physics) -> np.ndarray:
            angular_velocities = []
            for fingertip_site in self._entity.fingertip_sites:
                ang_vel = mujoco_utils.get_site_velocity(physics, fingertip_site)
                angular_velocities.append(ang_vel[3:])
            return np.concatenate(angular_velocities)

        return observable.Generic(
            raw_observation_callable=_get_fingertip_angular_velocities
        )
