from __future__ import annotations

import abc
import dataclasses
from typing import Optional, Sequence

import numpy as np
from dm_control.mujoco.wrapper import mjbindings

from shadow_hand import hints


mjlib = mjbindings.mjlib
enums = mjbindings.enums


class CartesianVelocitytoJointVelocityMapper(abc.ABC):
    """Abstract base class for a Cartesian velocity to joint velocity mapper.

    Subclasses should implement the `compute_joint_velocities` method, which maps a
    Cartesian 6D velocity about the global frame to joint velocities. The target
    Cartesian velocity is specified as a 6D vector, with the 3D linear velocity term
    followed by the 3D angular velocity term.

    At every call to `compute_joint_velocities`, the mapper will attempt to solve for
    the joint velocities that realize the target Cartesian velocity on the frame
    attached to the MuJoCo object defined by `object_type` and `object_name`. The target
    Cartesian velocity must be expressed about the MuJoCo object's origin, in world
    orientation. An error is returned if the computed velocities do not exist, or if the
    mapper fails to compute the velocities.

    This class expects an updated MjData object at every call to
    `compute_joint_velocities`, so users should ensure that the mjData object has
    up-to-date `qpos` and `qvel` fields.
    """

    @abc.abstractmethod
    def compute_joint_velocities(
        self,
        data: hints.MjData,
        target_velocities: Sequence[np.ndarray],
        nullspace_bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Maps the Cartesian target velocity to joint velocities."""


@dataclasses.dataclass
class Parameters:
    """Container for `CartesianVelocitytoJointVelocityMapper` parameters."""

    model: hints.MjModel
    """MuJoCo `MjModel` instance."""

    object_types: Sequence[hints.MujocoObjectType]
    """MuJoCo type of the object being controlled. Only bodies, geoms and sites are
    currently supported."""

    object_names: Sequence[str]
    """MuJoCo name of the object being controlled."""

    def validate_parameters(self) -> None:
        """Validates the parameters."""

        for object_type in self.object_types:
            if object_type not in [
                enums.mjtObj.mjOBJ_BODY,
                enums.mjtObj.mjOBJ_GEOM,
                enums.mjtObj.mjOBJ_SITE,
            ]:
                raise ValueError(
                    f"Objects of type {object_type} are not supported. Only"
                    " bodies, geoms and sites are supported."
                )

        for object_name, object_type in zip(
            self.object_names, self.object_types
        ):
            if self.model.name2id(object_name, object_type) < 0:
                raise ValueError(
                    f"Could not find MuJoCo object with name {object_name} and"
                    f" type {object_type} in the provided model."
                )
