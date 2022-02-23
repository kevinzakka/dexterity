import copy
import dataclasses
from typing import Mapping, Optional, Sequence

import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings
from dm_robotics.geometry import geometry, mujoco_physics

from shadow_hand import controllers
from shadow_hand.models.hands import shadow_hand_e_constants as consts
from shadow_hand.utils import geometry_utils, mujoco_utils


mjlib = mjbindings.mjlib

# Gain for the linear and angular twist computation, these values should always
# be between 0 and 1. 0 corresponds to not moving and 1 corresponds to moving to the
# target in a single integration timestep.
_LINEAR_VELOCITY_GAIN = 0.95
_ANGULAR_VELOCITY_GAIN = 0.95

_NULLSPACE_GAIN = 0.4

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

    def __init__(
        self,
        model: mjcf.RootElement,
        fingers: Sequence[consts.Components],
        prefix: str = "",
        nullspace_gain: float = _NULLSPACE_GAIN,
    ) -> None:
        """Constructor.

        Args:
            model: The MJCF model root.
            prefix: The prefix assigned to the hand model in case it is attached to
                another entity.
            nullspace_gain: Scales the nullspace velocity bias. If the gain is set to 0,
                there will be no nullspace optimization during the solve process.
        """
        self._nullspace_gain = nullspace_gain
        self._fingers = fingers
        self._physics = mjcf.Physics.from_mjcf_model(model)
        self._geometry_physics = mujoco_physics.wrap(self._physics)

        # Wrist information.
        wrist_joint_names = [
            j.name for j in consts.JOINT_GROUP[consts.Components.WR]
        ]
        self._wrist_controllable_joints = []
        for joint_name in wrist_joint_names:
            joint_elem = model.find(
                "joint", mujoco_utils.prefix_identifier(joint_name, prefix)
            )
            self._wrist_controllable_joints.append(joint_elem)
        assert len(self._wrist_controllable_joints) == len(wrist_joint_names)
        self._wrist_num_joints = len(wrist_joint_names)
        self._wirst_joint_bindings = self._physics.bind(
            self._wrist_controllable_joints
        )
        self._wrist_nullspace_joint_position_reference = np.zeros(
            self._wrist_num_joints
        )

        # Finger information.
        self._controllable_joints = {}
        self._elements = {}
        for finger in fingers:
            fingertip_name = consts.FINGER_FINGERTIP_MAPPING[finger]
            fingertip_site_name = mujoco_utils.prefix_identifier(
                f"{fingertip_name}_site", prefix
            )
            fingertip_site_elem = model.find("site", fingertip_site_name)
            assert fingertip_site_elem is not None
            self._elements[finger] = fingertip_site_elem

            joint_names = [j.name for j in consts.JOINT_GROUP[finger]]
            joint_names += wrist_joint_names
            controllable_joints = []
            for joint_name in joint_names:
                joint_elem = model.find(
                    "joint", mujoco_utils.prefix_identifier(joint_name, prefix)
                )
                controllable_joints.append(joint_elem)
            assert len(controllable_joints) == len(joint_names)
            self._controllable_joints[finger] = controllable_joints

        # Get all the joints of the hand.
        self._all_joints = []
        for joint_name in consts.JOINT_NAMES:
            joint_elem = model.find(
                "joint", mujoco_utils.prefix_identifier(joint_name, prefix)
            )
            assert joint_elem is not None
            self._all_joints.append(joint_elem)

        self._joint_bindings = {}
        self._num_joints = {}
        for finger, controllable_joints in self._controllable_joints.items():
            self._joint_bindings[finger] = self._physics.bind(
                controllable_joints
            )
            self._num_joints[finger] = len(controllable_joints)
        self._all_joints_binding = self._physics.bind(self._all_joints)

        self._nullspace_joint_position_reference = 0.5 * np.sum(
            self._all_joints_binding.range, axis=1
        )

        self._create_mapper()

    def _create_mapper(self) -> None:
        obj_types = []
        obj_names = []
        for finger in self._fingers:
            obj_types.append(
                mujoco_utils.get_element_type(self._elements[finger])
            )
            obj_names.append(self._elements[finger].full_identifier)
        params = controllers.dls.DampedLeastSquaresParameters(
            model=self._physics.model,
            object_types=obj_types,
            object_names=obj_names,
            regularization_weight=_REGULARIZATION_WEIGHT,
        )
        self._mapper = controllers.dls.DampedLeastSquaresMapper(params)

    def solve(
        self,
        target_positions: Mapping[consts.Components, np.ndarray],
        linear_tol: float = 1e-3,
        max_steps: int = 100,
        early_stop: bool = False,
        num_attempts: int = 30,
        stop_on_first_successful_attempt: bool = False,
    ) -> Optional[np.ndarray]:
        # Set the initial finger configuration to zero.
        inital_joint_configuration = {}
        for finger, num_joints in self._num_joints.items():
            inital_joint_configuration[finger] = np.zeros(num_joints)

        nullspace_jnt_qpos_min_err: float = np.inf
        success: bool = False
        sol_qpos: Optional[np.ndarray] = None

        # Each iteration of this loop attempts to solve the IK problem.
        for attempt in range(num_attempts):
            # Randomize the initial joint configuration so that the IK can find
            # different solutions.
            if attempt == 0:
                for finger, joint_binding in self._joint_bindings.items():
                    joint_binding.qpos[:] = inital_joint_configuration[finger]
            else:
                for finger, joint_binding in self._joint_bindings.items():
                    joint_binding.qpos[:] = np.random.uniform(
                        joint_binding.range[:, 0], joint_binding.range[:, 1]
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
                    np.linalg.norm(
                        solution.qpos - self._nullspace_joint_position_reference
                    )
                )
                if nullspace_jnt_qpos_err < nullspace_jnt_qpos_min_err:
                    nullspace_jnt_qpos_min_err = nullspace_jnt_qpos_err
                    sol_qpos = solution.qpos

            if success and stop_on_first_successful_attempt:
                break

        if not success:
            print(f"{self.__class__.__name__} failed to find a solution.")

        return sol_qpos

    def _solve_ik(
        self,
        target_positions: Mapping[consts.Components, np.ndarray],
        linear_tol: float,
        max_steps: int,
        early_stop: bool,
    ) -> _Solution:
        """Solves for a joint configuration that brings element pose to target pose."""
        cur_frames = {}
        cur_poses = {}
        previous_poses = {}
        for finger, target_position in target_positions.items():
            cur_frame = geometry.PoseStamped(
                pose=None, frame=self._elements[finger]
            )
            cur_pose = cur_frame.get_world_pose(self._geometry_physics)
            cur_frames[finger] = cur_frame
            cur_poses[finger] = cur_pose
            previous_poses[finger] = copy.copy(cur_pose)

        # Each iteration of this loop attempts to reduce the error between the site's
        # position and the target position.
        for _ in range(max_steps):
            twists = []
            for finger, target_position in target_positions.items():
                twist = _compute_twist(
                    cur_poses[finger],
                    geometry.Pose(position=target_position, quaternion=None),
                    _LINEAR_VELOCITY_GAIN,
                    _ANGULAR_VELOCITY_GAIN,
                    _INTEGRATION_TIMESTEP_SEC,
                )
                twists.append(twist.linear)

            qdot_sol = self._compute_joint_velocities(twists)

            mjbindings.mjlib.mj_integratePos(
                self._physics.model.ptr,
                self._physics.data.qpos,
                qdot_sol,
                _INTEGRATION_TIMESTEP_SEC,
            )
            self._update_physics_data()

            avg_linear_err: float = 0.0
            close_enough: bool = True
            not_enough_progress: bool = False

            for finger, target_position in target_positions.items():
                # Get the distance between the current pose and the target pose.
                cur_pose = cur_frames[finger].get_world_pose(
                    self._geometry_physics
                )
                linear_err = float(
                    np.linalg.norm(target_position - cur_pose.position)
                )
                avg_linear_err += linear_err

                # Stop if the pose is close enough to the target pose.
                if linear_err > linear_tol:
                    close_enough = False

                # Stop the solve if not enough progress is being made.
                previous_pose = previous_poses[finger]
                linear_change = np.linalg.norm(
                    cur_pose.position - previous_pose.position
                )
                if linear_err / (linear_change + 1e-10) > _PROGRESS_THRESHOLD:
                    not_enough_progress = True

                previous_poses[finger] = copy.copy(cur_pose)
                cur_poses[finger] = cur_pose

            # Average out the linear error.
            avg_linear_err /= len(target_positions)

            # Break conditions.
            if early_stop and close_enough:
                break
            if not_enough_progress:
                break

        qpos = np.array(self._physics.data.qpos)
        return _Solution(qpos=qpos, linear_err=avg_linear_err)

    def _compute_joint_velocities(
        self, cartesian_6d_target: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Maps a Cartesian 6D target velocity to joint velocities."""
        nullspace_bias = None
        if self._nullspace_gain > 0.0:
            nullspace_bias = (
                _NULLSPACE_GAIN
                * (
                    self._nullspace_joint_position_reference
                    - self._all_joints_binding.qpos
                )
                / _INTEGRATION_TIMESTEP_SEC
            )

        return self._mapper.compute_joint_velocities(
            data=self._physics.data,
            target_velocities=cartesian_6d_target,
            nullspace_bias=nullspace_bias,
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
        mjlib.mj_normalizeQuat(self._physics.model.ptr, self._physics.data.qpos)
        mjlib.mj_kinematics(self._physics.model.ptr, self._physics.data.ptr)
        mjlib.mj_comPos(self._physics.model.ptr, self._physics.data.ptr)


def _compute_twist(
    init_pose: geometry.Pose,
    final_pose: geometry.Pose,
    linear_velocity_gain: float,
    angular_velocity_gain: float,
    control_timestep_seconds: float,
) -> geometry.Twist:
    """Returns the twist to apply to the element to reach final_pose from init_pose."""
    position_error = final_pose.position - init_pose.position
    orientation_error = geometry_utils.get_orientation_error(
        to_quat=final_pose.quaternion,
        from_quat=init_pose.quaternion,
    )
    linear = linear_velocity_gain * position_error / control_timestep_seconds
    angular = (
        angular_velocity_gain * orientation_error / control_timestep_seconds
    )
    return geometry.Twist(np.concatenate((linear, angular)))
