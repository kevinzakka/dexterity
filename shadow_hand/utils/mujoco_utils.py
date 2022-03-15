from typing import List, Sequence

import mujoco
import numpy as np
from dm_control import mjcf
from dm_robotics.transformations import transformations as tr

from shadow_hand import hints


def prefix_identifier(identifier: str, prefix: str) -> str:
    """Prefixes the identifier with the provided string.

    This helper function deals with possibly empty prefixes and does not add the slash
    delimiter.
    """
    if not prefix:
        return identifier
    return f"{prefix}/{identifier}"


def get_site_pose(physics: mjcf.Physics, site_elem: hints.MjcfElement) -> np.ndarray:
    """Returns the world pose of the site as a 4x4 transform.

    Args:
        physics: An `mjcf.Physics` instance.
        site_entity: An `mjcf.Element` instance.
    """
    binding = physics.bind(site_elem)
    xpos = binding.xpos.reshape(3, 1)
    xmat = binding.xmat.reshape(3, 3)
    return np.vstack(
        [
            np.hstack([xmat, xpos]),
            np.array([0, 0, 0, 1]),
        ]
    )


def get_site_relative_pose(
    physics: mjcf.Physics, site_a: hints.MjcfElement, site_b: hints.MjcfElement
) -> np.ndarray:
    """Returns the pose of `site_a` in the frame of `site_b`.

    Args:
        physics: An `mjcf.Physics` instance.
        site_a: An `mjcf.Element` instance.
        site_b: An `mjcf.Element` instance.
    """
    pose_wa = get_site_pose(physics, site_a)  # Pose of site_a in world frame.
    pose_wb = get_site_pose(physics, site_b)  # Pose of site_b in world frame.
    pose_bw = tr.hmat_inv(pose_wb)  # Pose of world frame in site_b.
    pose_ba = pose_bw @ pose_wa  # Pose of site_a in site_b.
    return pose_ba


def get_site_velocity(
    physics: mjcf.Physics, site_elem: mjcf.Element, world_frame: bool = False
) -> np.ndarray:
    """Returns the linear and angular velocities of the site.

    This 6-D vector represents the instantaneous velocity of the coordinate system
    attached to the site. If `world_frame=True`, the velocity is expressed in the world
    frame.

    Args:
        physics: An `mjcf.Physics` instance.
        site_elem: An `mjcf.Element` instance.
        world_frame: Whether to return the velocity in the world frame.
    """
    flg_local = 0 if world_frame else 1
    idx = physics.model.name2id(site_elem.full_identifier, mujoco.mjtObj.mjOBJ_SITE)
    site_vel = np.empty(6)
    mujoco.mj_objectVelocity(
        physics.model.ptr,
        physics.data.ptr,
        mujoco.mjtObj.mjOBJ_SITE,
        idx,
        site_vel,
        flg_local,
    )
    return np.hstack([site_vel[3:], site_vel[:3]])


def get_joint_dof_size(model: hints.MjModel, joint_id: int) -> int:
    """Returns the number of degrees of freedom of the joint."""
    if joint_id < 0 or joint_id >= model.njnt:
        raise ValueError(
            f"Provided joint_id {joint_id} is invalid for the provided model, "
            f"which has {model.njnt} joints."
        )

    joint_type = model.jnt_type[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
        return 1
    elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return 1
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    else:  # Not supported.
        return -1


def joint_ids_to_dof_ids(model: hints.MjModel, joint_ids: Sequence[int]) -> List[int]:
    dof_ids = []
    for joint_id in joint_ids:
        dof = get_joint_dof_size(model, joint_id)
        if dof == -1:
            raise ValueError(f"joint_id {joint_id} is not a recognized joint type.")
        for i in range(dof):
            dof_ids.append(model.jnt_dofadr[joint_id] + i)
    return dof_ids


def compute_object_6d_jacobian(
    model: hints.MjModel,
    data: hints.MjData,
    object_type: hints.MujocoObjectType,
    object_id: int,
) -> np.ndarray:
    """Computes the (6, nv) object Jacobian.

    The Jacobian maps joint velocities to the object's Cartesian 6D velocity in the
    world frame. Each row of the Jacobian is the gradient of the corresponding 3D
    coordinate of the specified point with respect to the degrees of freedom.

    Only MuJoCo bodies, geoms and sites are supported.
    """
    jacobian = np.empty((6, model.nv), dtype=data.qpos.dtype)
    jacobian_position, jacobian_rotation = jacobian[:3], jacobian[3:]

    if object_type == mujoco.mjtObj.mjOBJ_BODY:
        func = mujoco.mj_jacBody
    elif object_type == mujoco.mjtObj.mjOBJ_GEOM:
        func = mujoco.mj_jacGeom
    elif object_type == mujoco.mjtObj.mjOBJ_SITE:
        func = mujoco.mj_jacSite
    else:
        raise ValueError(
            f"Invalid `object_type` {object_type}. Only bodies, geoms and sites"
            " are supported."
        )

    func(
        model.ptr,
        data.ptr,
        jacobian_position,
        jacobian_rotation,
        object_id,
    )

    return jacobian


def get_element_type(element: hints.MjcfElement) -> mujoco.mjtObj:
    if element.tag == "body":
        return mujoco.mjtObj.mjOBJ_BODY
    elif element.tag == "geom":
        return mujoco.mjtObj.mjOBJ_GEOM
    elif element.tag == "site":
        return mujoco.mjtObj.mjOBJ_SITE
    else:
        raise ValueError(
            f"Element must be a MuJoCo body, geom or site. Got [{element.tag}]."
        )


def compensate_gravity(
    physics: mjcf.Physics, body_elements: Sequence[hints.MjcfElement]
) -> None:
    """Counteracts gravity by applying forces to body elements."""
    gravity = np.hstack([physics.model.opt.gravity, [0.0, 0.0, 0.0]])
    bodies = physics.bind(body_elements)
    bodies.xfrc_applied = -gravity * bodies.mass[..., None]
