import dataclasses
from typing import Optional, Sequence

import numpy as np

from shadow_hand import hints
from shadow_hand.controllers import mapper
from shadow_hand.utils import mujoco_utils


@dataclasses.dataclass
class DampedLeastSquaresParameters(mapper.Parameters):

    regularization_weight: float
    """Damping factor used to regularize the pseudoinverse."""

    def validate_parameters(self) -> None:
        if self.regularization_weight < 0:
            raise ValueError(
                "`regularization_weight` must be non-negative, but was "
                f"{self.regularization_weight}."
            )

        super().validate_parameters()


@dataclasses.dataclass
class DampedLeastSquaresMapper(mapper.CartesianVelocitytoJointVelocityMapper):
    """A `CartesianVelocitytoJointVelocityMapper` that uses damped least-squares to
    solve for the joint velocities.

    Specifically, it maps the desired Cartesian velocity V to desired joint velocities
    v using the relationship:

    `J^T v = [J^T @ J + λI] @ V`

    where @ denotes the matrix product, J is the end-effector Jacobian, λ is a damping
    factor and I is the identity matrix.
    """

    params: DampedLeastSquaresParameters

    def __post_init__(self) -> None:
        self.params.validate_parameters()

    def compute_joint_velocities(
        self,
        data: hints.MjData,
        target_velocities: Sequence[np.ndarray],
        nullspace_bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        del nullspace_bias

        jacobians = []
        for obj_type, obj_name in zip(
            self.params.object_types, self.params.object_names
        ):
            jacobian = mujoco_utils.compute_object_6d_jacobian(
                model=self.params.model,
                data=data,
                object_type=obj_type,
                object_id=self.params.model.name2id(obj_name, obj_type),
            )
            jacobians.append(jacobian[:3])
        jacobian = np.concatenate(jacobians, axis=0)
        twist = np.concatenate(target_velocities, axis=0)

        # Solve!
        hess_approx = jacobian.T @ jacobian
        hess_approx += np.eye(hess_approx.shape[0]) * self.params.regularization_weight
        return np.linalg.solve(hess_approx, jacobian.T @ twist)
