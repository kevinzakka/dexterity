import abc
import dataclasses
import enum
from typing import Callable, List, Optional, Tuple

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_robotics.transformations import transformations as tr

from dexterity.hints import FloatArray
from dexterity.hints import MjcfElement
from dexterity.utils import mujoco_collisions
from dexterity.utils import mujoco_utils

_DEFAULT_XPOS = (0.0, 0.0, 0.0)
_DEFAULT_XQUAT = (1.0, 0.0, 0.0, 0.0)


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


def _make_readonly_float64_copy(value: FloatArray) -> np.ndarray:
    out = np.array(value, dtype=np.float64)
    out.flags.writeable = False
    return out


@dataclasses.dataclass(frozen=True)
class HandPose:
    """A container for a hand's joint and Cartesian pose."""

    qpos: Optional[np.ndarray] = None
    xpos: np.ndarray = _make_readonly_float64_copy(_DEFAULT_XPOS)
    xquat: np.ndarray = _make_readonly_float64_copy(_DEFAULT_XQUAT)

    @staticmethod
    def create(
        xpos: FloatArray,
        xquat: FloatArray,
        qpos: Optional[FloatArray] = None,
    ) -> "HandPose":
        return HandPose(
            xpos=_make_readonly_float64_copy(xpos),
            xquat=_make_readonly_float64_copy(xquat),
            qpos=None if qpos is None else _make_readonly_float64_copy(qpos),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandPose):
            return NotImplemented
        qpos_eq = bool(np.all(self.qpos == other.qpos))
        xpos_eq = bool(np.all(self.xpos == other.xpos))
        xquat_eq = bool(np.all(self.xquat == other.xquat))
        return qpos_eq and xpos_eq and xquat_eq


class DexterousHand(composer.Entity, abc.ABC):
    """Base composer class for a dexterous multi-fingered hand."""

    def _build(self) -> None:
        self._fingers_pos_sensors: Tuple[MjcfElement, ...] = ()

    def _build_observables(self) -> composer.Observables:
        return DexterousHandObservables(self)

    def _postprocess_sampled_joint_angles(self, qpos: np.ndarray) -> np.ndarray:
        """Post-process a joint configuration that has been randomly sampled.

        Underactuated hands might need to override this method.
        """
        return qpos

    @property
    def palm_upright_pose(self) -> HandPose:
        return HandPose()

    @property
    def num_joints(self) -> int:
        """Returns the number of joints (aka degrees of freedom) in the hand."""
        return len(self._joints)

    dofs = num_joints

    @property
    def num_actuators(self) -> int:
        """Returns the number of actuators in the hand."""
        return len(self._actuators)

    @property
    def underactuated(self) -> bool:
        """Returns True if the hand has less actuators than degrees of freedom."""
        return self.num_actuators < self.num_joints

    @property
    def fingers_pos_sensors(self) -> Tuple[MjcfElement, ...]:
        return self._fingers_pos_sensors

    @property
    def tendons(self) -> Tuple[MjcfElement, ...]:
        raise NotImplementedError

    def sample_joint_angles(
        self,
        physics: mjcf.Physics,
        random_state: np.random.RandomState,
        range_fraction: float = 1.0,
    ) -> np.ndarray:
        """Samples a random joint configuration for the hand.

        This is not guaranteed to be collision-free. If you need a collision-free
        configuration, use `sample_collision_free_joint_angles` instead.

        Args:
            physics: An `mjcf.Physics` instance.
            random_state: A `np.random.RandomState` instance.
            range_fraction: What fraction of the joint's total range to sample from.
                Defaults to 1.0 which means that the full range will be used.
        """
        if not 0 <= range_fraction <= 1:
            raise ValueError("range_fraction must be between 0 and 1.")

        lower, upper = (range_fraction * physics.bind(self.joints).range).T
        qpos = random_state.uniform(lower, upper)
        return self._postprocess_sampled_joint_angles(qpos)

    def sample_collision_free_joint_angles(
        self,
        physics: mjcf.Physics,
        random_state: np.random.RandomState,
        range_fraction: float = 1.0,
    ) -> np.ndarray:
        """Samples a collision-free joint configuration.

        Args:
            physics: An `mjcf.Physics` instance.
            random_state: A `np.random.RandomState` instance.
            range_fraction: What fraction of each joint's total range to sample from.
                Defaults to 1.0 which means that the full range will be used.
        """
        # Note: We're making a copy of the physics object so that whatever changes we
        # make to, for example the joint angles, do not affect the original physics
        # instance.
        physics = physics.copy()
        while True:
            qpos = self.sample_joint_angles(physics, random_state, range_fraction)
            self.set_joint_angles(physics, qpos)
            physics.forward()
            if not mujoco_collisions.has_self_collision(physics, self.name):
                break
        return qpos

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
    def root_body(self) -> MjcfElement:
        """Returns the root body of the hand.

        Egocentric observations will be computed relative to this body.
        """

    @property
    @abc.abstractmethod
    def bodies(self) -> Tuple[MjcfElement, ...]:
        """List of bodies belonging to the hand."""

    @property
    @abc.abstractmethod
    def joints(self) -> Tuple[MjcfElement, ...]:
        """List of joint elements belonging to the hand."""

    @property
    @abc.abstractmethod
    def actuators(self) -> Tuple[MjcfElement, ...]:
        """List of actuator elements belonging to the hand."""

    @property
    @abc.abstractmethod
    def fingertip_sites(self) -> Tuple[MjcfElement, ...]:
        """List of fingertip site elements belonging to the hand."""

    @property
    @abc.abstractmethod
    def joint_torque_sensors(self) -> Tuple[MjcfElement, ...]:
        """List of joint torque sensor elements belonging to the hand."""

    @property
    @abc.abstractmethod
    def joint_groups(self) -> Tuple[JointGrouping, ...]:
        """A list of `JointGrouping` objects corresponding to hand parts."""

    # Abstract methods.

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

    @abc.abstractmethod
    def set_joint_angles(self, physics: mjcf.Physics, qpos: np.ndarray) -> None:
        """Sets the joints of the hand to a given configuration.

        This method is abstract because a hand might have to perform extra logic
        specific to its actuation after setting the joint angles.
        """


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
            return np.vstack([np.sin(qpos), np.cos(qpos)]).T.ravel()

        return observable.Generic(raw_observation_callable=_get_joint_pos_sin_cos)

    @composer.observable
    def joint_velocities(self) -> observable.MJCFFeature:
        return observable.MJCFFeature(kind="qvel", mjcf_element=self._entity.joints)

    @composer.observable
    def joint_torques(self) -> observable.Generic:
        def _get_joint_torques(physics: mjcf.Physics) -> np.ndarray:
            # We only care about torques acting on each joint's axis of rotation, so we
            # project them.
            torques = physics.bind(self._entity.joint_torque_sensors).sensordata
            joint_axes = physics.bind(self._entity.joints).axis
            return np.einsum("ij,ij->i", torques.reshape(-1, 3), joint_axes)

        return observable.Generic(raw_observation_callable=_get_joint_torques)

    @composer.observable
    def fingertip_positions(self) -> observable.Generic:
        """3D position of the fingertips relative to the world frame."""

        def _get_fingertip_positions(physics: mjcf.Physics) -> np.ndarray:
            return physics.bind(self._entity.fingertip_sites).xpos.ravel()

        return observable.Generic(raw_observation_callable=_get_fingertip_positions)

    @composer.observable
    def fingertip_orientations(self) -> observable.Generic:
        """3D orientation of the fingertips relative to the world frame."""

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
                lin_vel = mujoco_utils.get_site_velocity(
                    physics, fingertip_site, world_frame=True
                )
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
                ang_vel = mujoco_utils.get_site_velocity(
                    physics, fingertip_site, world_frame=True
                )
                angular_velocities.append(ang_vel[3:])
            return np.concatenate(angular_velocities)

        return observable.Generic(
            raw_observation_callable=_get_fingertip_angular_velocities
        )

    @composer.observable
    def fingertip_positions_ego(self):
        """3D position of the fingers, relative to the root, in the egocentric frame."""

        fingers_pos_sensors = []
        for fingertip_site in self._entity.fingertip_sites:
            fingers_pos_sensors.append(
                self._entity.mjcf_model.sensor.add(
                    "framepos",
                    name=fingertip_site.name + "_pos_sensor",
                    objtype=fingertip_site.tag,
                    objname=fingertip_site.name,
                    reftype="body",
                    refname=self._entity.root_body,
                )
            )
        self._entity._fingers_pos_sensors = fingers_pos_sensors

        def _get_relative_pos_in_egocentric_frame(physics: mjcf.Physics) -> np.ndarray:
            return np.reshape(
                physics.bind(self._entity.fingers_pos_sensors).sensordata, -1
            )

        return observable.Generic(_get_relative_pos_in_egocentric_frame)

    # Semantic grouping of the above observables.

    def _collect_from_attachments(
        self, attribute_name: str
    ) -> List[Callable[..., np.ndarray]]:
        out: List[Callable[..., np.ndarray]] = []
        for entity in self._entity.iter_entities(exclude_self=True):
            out.extend(getattr(entity.observables, attribute_name, []))
        return out

    @property
    def proprioception(self) -> List[Callable[..., np.ndarray]]:
        return [
            self.joint_positions,
            self.joint_positions_sin_cos,
            self.joint_torques,
            self.fingertip_positions,
            self.fingertip_orientations,
            self.fingertip_linear_velocities,
            self.fingertip_angular_velocities,
        ] + self._collect_from_attachments("proprioception")
