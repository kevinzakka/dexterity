# TODO(kevin): In the future, we'd like to solve for finger orientation as well.

import copy
import dataclasses
from typing import List, Optional, Sequence

import mujoco
import numpy as np
from absl import logging
from dm_control import mjcf
from dm_robotics.geometry import geometry
from dm_robotics.geometry import mujoco_physics

from dexterity import controllers
from dexterity.models.hands import dexterous_hand
from dexterity.utils import mujoco_utils

# Gain for the linear twist computation, should always be between 0 and 1.
# 0 corresponds to not moving and 1 corresponds to moving to the target in a single
# integration timestep.
_LINEAR_VELOCITY_GAIN = 0.95

# Integration timestep used to convert from joint velocities to joint positions.
_INTEGRATION_TIMESTEP_SEC = 1.0

# Damping factor.
_REGULARIZATION_WEIGHT = 1e-5

# If the norm of the error divided by the magnitude of the joint position update is
# greater than this value, then the solve is ended prematurely. This helps us avoid
# getting stuck in local minima.
_PROGRESS_THRESHOLD = 20.0


@dataclasses.dataclass
class _Solution:
    """Return value of an IK solution."""

    qpos: np.ndarray
    linear_err: float


class IKSolver:
    """Inverse kinematics solver for a dexterous hand."""

    def __init__(self, hand: dexterous_hand.DexterousHand) -> None:
        # Note: We need the root model in case the hand is attached to another entity.
        self._physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model.root_model)
        self._geometry_physics = mujoco_physics.wrap(self._physics)

        # Get joint bindings.
        self._elements = hand.fingertip_sites
        self._all_joints_binding = self._physics.bind(hand.joints)
        self._joint_bindings = []
        for joint_group in hand.joint_groups:
            joint_binding = self._physics.bind(joint_group.joints)
            self._joint_bindings.append(joint_binding)

        # Make the midrange of the joints be the nullspace.
        self._nullspace_reference = np.mean(self._all_joints_binding.range, axis=1)

        self._create_mapper()

    def _create_mapper(self) -> None:
        obj_types = [
            mujoco_utils.get_element_type(element) for element in self._elements
        ]
        obj_names = [element.full_identifier for element in self._elements]
        params = controllers.dls.DampedLeastSquaresParameters(
            model=self._physics.model,
            object_types=obj_types,
            object_names=obj_names,
            regularization_weight=_REGULARIZATION_WEIGHT,
        )
        self._mapper = controllers.dls.DampedLeastSquaresMapper(params)

    def solve(
        self,
        target_positions: np.ndarray,
        linear_tol: float = 1e-3,
        max_steps: int = 100,
        early_stop: bool = False,
        num_attempts: int = 30,
        stop_on_first_successful_attempt: bool = False,
    ) -> Optional[np.ndarray]:
        target_positions = target_positions.reshape(-1, 3)
        if target_positions.shape[0] != len(self._elements):
            raise ValueError(
                "The number of target positions must be equal to the number of "
                "end-effector sites."
            )

        nullspace_jnt_qpos_min_err: float = np.inf
        success: bool = False
        sol_qpos: Optional[np.ndarray] = None

        # Each iteration of this loop attempts to solve the IK problem.
        for attempt in range(num_attempts):
            # Randomize the initial joint configuration so that the IK can find
            # different solutions.
            if attempt == 0:
                self._all_joints_binding.qpos = self._nullspace_reference
            else:
                self._all_joints_binding.qpos = np.random.uniform(
                    *self._all_joints_binding.range.T
                )

            solution = self._solve_ik(
                target_positions,
                linear_tol,
                max_steps,
                early_stop,
            )

            if solution.linear_err <= linear_tol:
                success = True

                nullspace_jnt_qpos_err = float(
                    np.linalg.norm(solution.qpos - self._nullspace_reference)
                )
                if nullspace_jnt_qpos_err < nullspace_jnt_qpos_min_err:
                    nullspace_jnt_qpos_min_err = nullspace_jnt_qpos_err
                    sol_qpos = solution.qpos

            if success and stop_on_first_successful_attempt:
                break

        if not success:
            logging.warning(f"{self.__class__.__name__} failed to find a solution.")

        return sol_qpos

    def _solve_ik(
        self,
        target_positions: np.ndarray,
        linear_tol: float,
        max_steps: int,
        early_stop: bool,
    ) -> _Solution:
        """Solves for a joint configuration that brings element pose to target pose."""
        cur_frames: List[geometry.PoseStamped] = []
        cur_poses: List[geometry.Pose] = []
        previous_poses: List[geometry.Pose] = []
        for element in self._elements:
            cur_frame = geometry.PoseStamped(pose=None, frame=element)
            cur_pose = cur_frame.get_world_pose(self._geometry_physics)
            cur_frames.append(cur_frame)
            cur_poses.append(cur_pose)
            previous_poses.append(copy.copy(cur_pose))

        # Each iteration of this loop attempts to reduce the error between the site's
        # position and the target position.
        for _ in range(max_steps):
            twists = []
            for i, target_position in enumerate(target_positions):
                twist = _compute_twist(
                    cur_poses[i],
                    geometry.Pose(position=target_position, quaternion=None),
                    _LINEAR_VELOCITY_GAIN,
                    _INTEGRATION_TIMESTEP_SEC,
                )
                twists.append(twist)

            qdot_sol = self._compute_joint_velocities(twists)

            mujoco.mj_integratePos(
                self._physics.model.ptr,
                self._physics.data.qpos,
                qdot_sol,
                _INTEGRATION_TIMESTEP_SEC,
            )
            self._update_physics_data()

            avg_linear_err: float = 0.0
            close_enough: bool = True
            not_enough_progress: bool = False

            for i, target_position in enumerate(target_positions):
                # Get the distance between the current pose and the target pose.
                cur_pose = cur_frames[i].get_world_pose(self._geometry_physics)
                linear_err = float(np.linalg.norm(target_position - cur_pose.position))
                avg_linear_err += linear_err

                # Stop if the pose is close enough to the target pose.
                if linear_err > linear_tol:
                    close_enough = False

                # Stop the solve if not enough progress is being made.
                previous_pose = previous_poses[i]
                linear_change = np.linalg.norm(
                    cur_pose.position - previous_pose.position
                )
                if linear_err / (linear_change + 1e-10) > _PROGRESS_THRESHOLD:
                    not_enough_progress = True

                previous_poses[i] = copy.copy(cur_pose)
                cur_poses[i] = cur_pose

            # Average out the linear error.
            avg_linear_err /= len(target_positions)

            # Break conditions.
            if early_stop and close_enough:
                break
            if not_enough_progress:
                break

        qpos = np.array(self._all_joints_binding.qpos, copy=True)
        return _Solution(qpos=qpos, linear_err=avg_linear_err)

    def _compute_joint_velocities(
        self, cartesian_6d_target: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Maps a Cartesian 6D target velocity to joint velocities."""
        return self._mapper.compute_joint_velocities(
            data=self._physics.data,
            target_velocities=cartesian_6d_target,
            nullspace_bias=None,
        )

    def _update_physics_data(self) -> None:
        """Updates the physics data following the integration of velocities."""
        # Clip joint positions.
        qpos = self._all_joints_binding.qpos
        min_range = self._all_joints_binding.range[:, 0]
        max_range = self._all_joints_binding.range[:, 1]
        qpos = np.clip(qpos, min_range, max_range)
        self._all_joints_binding.qpos[:] = qpos

        # Forward kinematics to update the pose of the tracked site.
        mujoco.mj_normalizeQuat(self._physics.model.ptr, self._physics.data.qpos)
        mujoco.mj_kinematics(self._physics.model.ptr, self._physics.data.ptr)
        mujoco.mj_comPos(self._physics.model.ptr, self._physics.data.ptr)


def _compute_twist(
    init_pose: geometry.Pose,
    final_pose: geometry.Pose,
    linear_velocity_gain: float,
    control_timestep_seconds: float,
) -> np.ndarray:
    """Returns the twist to apply to the element to reach final_pose from init_pose."""
    position_error = final_pose.position - init_pose.position
    linear = linear_velocity_gain * position_error / control_timestep_seconds
    return linear
